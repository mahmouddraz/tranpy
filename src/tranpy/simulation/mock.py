"""Mock simulation engine: a classical multi-machine model integrated with NumPy.

It replaces PowerFactory in tests, examples and method development. It is *not* a model
of the real IEEE systems: the network is a synthetic, seeded graph with the system's
number of generators and buses. What matters is that the stability label follows from
the simulated dynamics, as in a real simulator, instead of being drawn first and the
features generated from it.

Model (per unit, swing equations of the classical generator model)::

    d(delta_i)/dt = 2*pi*f0 * (omega_i - 1)
    2*H_i * d(omega_i)/dt = Pm_i - Pe_i - D_i * (omega_i - 1)
    Pe_i = sum_j E_i * E_j * B_ij * sin(delta_i - delta_j)

A fault on line (a, b) depresses the electrical power of nearby machines while it lasts;
clearing opens the line (B_ab = 0). Bus voltages are weighted phasor sums of the
generator EMFs. A machine is out of step once its angle relative to the centre of
inertia exceeds 180 degrees (latched), which defines the label.

All scenarios are integrated together (fixed-step RK4), so generating hundreds of events
takes seconds.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from tranpy.simulation.engine import FaultScenario, SimulationResult
from tranpy.systems import PowerSystem, get_system
from tranpy.vocabulary import Variable
from tranpy_core.errors import ConfigurationError
from tranpy_core.random import Seed, make_rng

__all__ = ["MockEngine", "MockNetwork"]

FloatArray = npt.NDArray[np.float64]

_BASE_MVA = 100.0
_FAULT_IMPEDANCE_REF_OHM = 0.05
_DEFAULT_ANGLE_SPREAD_RAD = {"ieee_9bus": 0.25, "new_england": 0.15}


@dataclass(frozen=True, eq=False)
class MockNetwork:
    """Parameters of the synthetic network (all arrays read-only)."""

    inertia_s: FloatArray
    damping_pu: FloatArray
    emf_pu: FloatArray
    coupling_pu: FloatArray
    lines: tuple[tuple[int, int], ...]
    initial_angle_rad: FloatArray
    mechanical_power_pu: FloatArray
    bus_weights: FloatArray
    load_generator: tuple[int, ...]
    load_base_pu: float = 0.5

    @classmethod
    def build(
        cls,
        system: PowerSystem,
        seed: Seed,
        *,
        coupling_scale: float = 1.0,
        angle_spread_rad: float = 0.25,
    ) -> MockNetwork:
        """Seeded random network with the size of ``system``."""
        rng = make_rng(seed)
        g, b = system.n_generators, system.n_buses
        edges = {tuple(sorted((i, (i + 1) % g))) for i in range(g)}
        target = max(g, round(1.5 * g))
        while len(edges) < min(target, g * (g - 1) // 2):
            i, j = rng.choice(g, size=2, replace=False)
            edges.add((int(min(i, j)), int(max(i, j))))
        lines: tuple[tuple[int, int], ...] = tuple(sorted((int(a), int(b)) for a, b in edges))
        coupling = np.zeros((g, g))
        for i, j in lines:
            coupling[i, j] = coupling[j, i] = rng.uniform(1.0, 3.0) * coupling_scale
        angles = rng.normal(0.0, angle_spread_rad, size=g)
        angles -= angles.mean()
        emf = rng.uniform(1.0, 1.1, size=g)
        pm = emf * (coupling * emf[None, :] * np.sin(angles[:, None] - angles[None, :])).sum(axis=1)
        weights = rng.dirichlet(np.full(g, 0.3), size=b)
        n_loads = max(3, b // 3)
        net = cls(
            inertia_s=rng.uniform(3.0, 6.0, size=g),
            damping_pu=np.full(g, 2.0),
            emf_pu=emf,
            coupling_pu=coupling,
            lines=lines,
            initial_angle_rad=angles,
            mechanical_power_pu=pm,
            bus_weights=weights,
            load_generator=tuple(int(x) for x in rng.integers(0, g, size=n_loads)),
        )
        for name in (
            "inertia_s",
            "damping_pu",
            "emf_pu",
            "coupling_pu",
            "initial_angle_rad",
            "mechanical_power_pu",
            "bus_weights",
        ):
            getattr(net, name).setflags(write=False)
        return net

    @property
    def n_generators(self) -> int:
        return int(self.emf_pu.shape[0])

    @property
    def n_buses(self) -> int:
        return int(self.bus_weights.shape[0])


@dataclass(frozen=True)
class MockEngine:
    """Always-available :class:`~tranpy.simulation.engine.SimulationEngine`.

    Attributes:
        system_name: system whose size the synthetic network copies.
        network_seed: seed of the network parameters (fixed per system by default, so
            every scenario of a dataset runs on the same network).
        simulation_time_s: end of the simulation; recording starts at ``-0.1`` s.
        step_s: integration and output step.
        coupling_scale: multiplies all line couplings.
        angle_spread_rad: spread of the pre-fault rotor angles, i.e. how heavily the
            network is loaded (larger = less stable). Default per system: 0.25 for the
            9-bus and 0.15 for the 39-bus network, which gives roughly 35 % unstable events
            at 20 cycles (9-bus) and 25 % at 12 cycles (39-bus) clearing time, and no
            unstable events for very short faults.
    """

    system_name: str = "ieee_9bus"
    network_seed: int | None = None
    simulation_time_s: float = 10.0
    step_s: float = 0.01
    coupling_scale: float = 1.0
    angle_spread_rad: float | None = None
    _network: MockNetwork = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.step_s <= 0 or self.simulation_time_s <= self.step_s:
            raise ConfigurationError("need 0 < step_s < simulation_time_s")
        seed = self.network_seed
        if seed is None:
            seed = zlib.crc32(self.system.key.encode())
        spread = self.angle_spread_rad
        if spread is None:
            spread = _DEFAULT_ANGLE_SPREAD_RAD.get(self.system.key, 0.2)
        network = MockNetwork.build(
            self.system, seed, coupling_scale=self.coupling_scale, angle_spread_rad=spread
        )
        object.__setattr__(self, "_network", network)

    @property
    def system(self) -> PowerSystem:
        return get_system(self.system_name)

    @property
    def network(self) -> MockNetwork:
        return self._network

    @property
    def n_lines(self) -> int:
        return len(self._network.lines)

    @property
    def n_loads(self) -> int:
        return len(self._network.load_generator)

    @property
    def channel_names(self) -> tuple[str, ...]:
        net = self._network
        width = max(2, len(str(max(net.n_buses, net.n_generators))))
        names = []
        for k in range(net.n_buses):
            bus = f"Bus {k + 1:0{width}d}"
            names += [
                f"{bus}|{Variable.BUS_VOLTAGE_PU.value}",
                f"{bus}|{Variable.BUS_ANGLE_DEG.value}",
                f"{bus}|{Variable.BUS_FREQUENCY_HZ.value}",
            ]
        for i in range(net.n_generators):
            gen = f"G {i + 1:0{width}d}"
            names += [
                f"{gen}|{v.value}"
                for v in (
                    Variable.ROTOR_SPEED_PU,
                    Variable.ACTIVE_POWER_MW,
                    Variable.REACTIVE_POWER_MVAR,
                    Variable.OUT_OF_STEP,
                    Variable.ROTOR_ANGLE_DEG,
                    Variable.ROTOR_FREQUENCY_DEVIATION_HZ,
                )
            ]
        return tuple(names)

    def simulate(self, scenarios: Sequence[FaultScenario]) -> list[SimulationResult]:
        """Integrate all scenarios together and return one result per scenario."""
        net = self._network
        s, g = len(scenarios), net.n_generators
        for sc in scenarios:
            bad = [
                i
                for i in (sc.fault_line, sc.trip_line)
                if i is not None and not 0 <= i < self.n_lines
            ]
            if bad or not 0 <= sc.load_index < self.n_loads:
                raise ConfigurationError(
                    f"scenario {sc.scenario_id} refers to elements outside the network"
                )
        f0 = self.system.nominal_frequency_hz
        omega_s = 2 * np.pi * f0
        time = np.round(np.arange(-0.1, self.simulation_time_s + self.step_s / 2, self.step_s), 10)

        start = np.array([sc.fault_start_s for sc in scenarios])
        clear = np.array([sc.fault_clear_s for sc in scenarios])
        trip_at = np.array(
            [
                sc.fault_clear_s + sc.trip_delay_s if sc.trip_line is not None else np.inf
                for sc in scenarios
            ]
        )
        z_fault = np.hypot(
            [sc.fault_resistance_ohm for sc in scenarios],
            [sc.fault_reactance_ohm for sc in scenarios],
        )
        depth = 1.0 / (1.0 + z_fault / _FAULT_IMPEDANCE_REF_OHM)

        b_pre = net.coupling_pu
        b_post = np.repeat(b_pre[None], s, axis=0)
        b_trip = b_post.copy()
        near_gen = np.zeros((s, g))
        near_bus = np.zeros((s, net.n_buses))
        for k, sc in enumerate(scenarios):
            a, b = net.lines[sc.fault_line]
            b_post[k, a, b] = b_post[k, b, a] = 0.0
            b_trip[k] = b_post[k]
            if sc.trip_line is not None:
                c, d = net.lines[sc.trip_line]
                b_trip[k, c, d] = b_trip[k, d, c] = 0.0
            near_gen[k] = 0.3 * (b_pre[a] + b_pre[b]) / np.max(b_pre)
            near_gen[k, [a, b]] = 1.0
            near_bus[k] = net.bus_weights[:, a] + net.bus_weights[:, b]
        pm = np.repeat(net.mechanical_power_pu[None], s, axis=0)
        for k, sc in enumerate(scenarios):
            pm[k, net.load_generator[sc.load_index]] -= (
                sc.load_change_pct / 100.0 * net.load_base_pu
            )

        emf, h, damping = net.emf_pu, net.inertia_s, net.damping_pu
        ee = emf[:, None] * emf[None, :]

        def regime(t: float) -> tuple[FloatArray, FloatArray]:
            on = (t >= start) & (t < clear)
            coupling = np.where(
                (t >= trip_at)[:, None, None],
                b_trip,
                np.where((t >= clear)[:, None, None], b_post, b_pre),
            )
            scale = 1.0 - (depth * on)[:, None] * np.minimum(near_gen, 1.0)
            return coupling, scale

        def derivatives(
            delta: FloatArray, omega: FloatArray, coupling: FloatArray, scale: FloatArray
        ) -> tuple[FloatArray, FloatArray]:
            diff = delta[:, :, None] - delta[:, None, :]
            pe = (coupling * ee * np.sin(diff)).sum(axis=2) * scale
            return omega_s * (omega - 1.0), (pm - pe - damping * (omega - 1.0)) / (2.0 * h)

        n_t = time.shape[0]
        delta = np.empty((n_t, s, g))
        omega = np.empty((n_t, s, g))
        pe_rec = np.empty((n_t, s, g))
        delta[0] = net.initial_angle_rad
        omega[0] = 1.0
        dt = self.step_s
        for k in range(n_t):
            coupling, scale = regime(time[k])
            d0, w0 = delta[k], omega[k]
            pe_rec[k] = (coupling * ee * np.sin(d0[:, :, None] - d0[:, None, :])).sum(
                axis=2
            ) * scale
            if k == n_t - 1:
                break
            k1d, k1w = derivatives(d0, w0, coupling, scale)
            k2d, k2w = derivatives(d0 + dt / 2 * k1d, w0 + dt / 2 * k1w, coupling, scale)
            k3d, k3w = derivatives(d0 + dt / 2 * k2d, w0 + dt / 2 * k2w, coupling, scale)
            k4d, k4w = derivatives(d0 + dt * k3d, w0 + dt * k3w, coupling, scale)
            delta[k + 1] = d0 + dt / 6 * (k1d + 2 * k2d + 2 * k3d + k4d)
            omega[k + 1] = w0 + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)

        coi = (delta * h).sum(axis=2, keepdims=True) / h.sum()
        relative = delta - coi
        out_of_step = np.maximum.accumulate(np.abs(relative) > np.pi, axis=0).astype(np.float64)
        phasor = np.einsum("bg,tsg->tsb", net.bus_weights, emf * np.exp(1j * delta))
        fault_on = (time[:, None] >= start[None]) & (time[:, None] < clear[None])
        dip = 1.0 - (depth[None, :, None] * fault_on[:, :, None]) * np.minimum(near_bus[None], 1.0)
        voltage = np.abs(phasor) * dip
        bus_angle = np.degrees(np.angle(phasor))
        bus_freq = f0 * np.einsum("bg,tsg->tsb", net.bus_weights, omega)
        reactive = emf * (emf * net.coupling_pu.sum(axis=1)) - (
            ee[None, None] * net.coupling_pu * np.cos(delta[..., :, None] - delta[..., None, :])
        ).sum(axis=3)

        bus_block = np.stack([voltage, bus_angle, bus_freq], axis=3).reshape(n_t, s, -1)
        gen_block = np.stack(
            [
                omega,
                pe_rec * _BASE_MVA,
                reactive * _BASE_MVA,
                out_of_step,
                np.degrees(relative),
                (omega - 1.0) * f0,
            ],
            axis=3,
        ).reshape(n_t, s, -1)
        values = np.concatenate([bus_block, gen_block], axis=2)
        names = self.channel_names
        return [
            SimulationResult(
                scenario=sc,
                time=time.copy(),
                values=np.ascontiguousarray(values[:, k]),
                channel_names=names,
            )
            for k, sc in enumerate(scenarios)
        ]
