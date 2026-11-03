"""Admittance matrices reduced to generator internal nodes (shared by EAC and TEF).

Classical model: each generator is a constant EMF ``E'`` behind its transient reactance
``x'_d``, loads are constant admittances. Adding one internal node per generator and
eliminating every network bus (Kron reduction) gives ``Y'``, the matrix of paper Eq. (1):
``Pe_i = sum_j |E_i||E_j| |Y'_ij| cos(delta_i - delta_j - theta_ij)``.

All quantities are per unit on one system base.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tranpy_core.errors import ConfigurationError, DataError

__all__ = [
    "Branches",
    "admittance_matrix",
    "internal_node_admittance",
    "kron_reduce",
    "with_shunt",
]

ComplexArray = npt.NDArray[np.complex128]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Branches:
    """Pi-model branches, buses numbered from 0.

    Attributes:
        from_bus, to_bus: terminal buses.
        r_pu, x_pu: series impedance.
        b_pu: total line charging (half at each end).
        tap: off-nominal ratio on the ``from_bus`` side (1 for lines).
    """

    from_bus: IntArray
    to_bus: IntArray
    r_pu: FloatArray
    x_pu: FloatArray
    b_pu: FloatArray
    tap: FloatArray

    def __post_init__(self) -> None:
        n = len(self.from_bus)
        if any(len(a) != n for a in (self.to_bus, self.r_pu, self.x_pu, self.b_pu, self.tap)):
            raise ConfigurationError("branch arrays must have equal length")
        if np.any(np.hypot(self.r_pu, self.x_pu) <= 0):
            raise ConfigurationError("branch impedance must be nonzero")
        if np.any(np.asarray(self.tap) <= 0):
            raise ConfigurationError("tap ratio must be positive")


def admittance_matrix(n_buses: int, branches: Branches) -> ComplexArray:
    """Bus admittance matrix of the pi-model branches."""
    y = np.zeros((n_buses, n_buses), dtype=np.complex128)
    for f, t, r, x, b, a in zip(
        branches.from_bus,
        branches.to_bus,
        branches.r_pu,
        branches.x_pu,
        branches.b_pu,
        branches.tap,
        strict=True,
    ):
        ys = 1.0 / complex(r, x)
        half = 0.5j * b
        y[f, f] += (ys + half) / a**2
        y[t, t] += ys + half
        y[f, t] -= ys / a
        y[t, f] -= ys / a
    return y


def with_shunt(y: ComplexArray, bus: int, admittance: complex) -> ComplexArray:
    """Copy of ``y`` with a shunt admittance at ``bus`` (a fault: ``1 / z_fault``)."""
    out = y.copy()
    out[bus, bus] += admittance
    return out


def kron_reduce(y: ComplexArray, keep: npt.ArrayLike) -> ComplexArray:
    """Eliminate every node not in ``keep``: ``Y_kk - Y_ke Y_ee^-1 Y_ek``."""
    keep_idx = np.asarray(keep, dtype=np.int64)
    drop = np.setdiff1d(np.arange(y.shape[0]), keep_idx)
    if drop.size == 0:
        return y[np.ix_(keep_idx, keep_idx)].copy()
    y_ee = y[np.ix_(drop, drop)]
    if np.linalg.cond(y_ee) > 1e12:
        raise DataError("eliminated part of the network is singular (an isolated bus?)")
    return y[np.ix_(keep_idx, keep_idx)] - y[np.ix_(keep_idx, drop)] @ np.linalg.solve(
        y_ee, y[np.ix_(drop, keep_idx)]
    )


def internal_node_admittance(
    y_bus: ComplexArray,
    generator_bus: npt.ArrayLike,
    xd1_pu: npt.ArrayLike,
    load_admittance: npt.ArrayLike | None = None,
) -> ComplexArray:
    """``Y'`` between generator internal nodes, generator ``g`` at internal node ``g``.

    Args:
        y_bus: bus admittance matrix of the network (with any fault shunt already added).
        generator_bus: terminal bus of each generator.
        xd1_pu: transient reactance of each generator on the system base.
        load_admittance: constant load admittance per bus (``conj(S) / |V|^2``), or None.
    """
    gen_bus = np.asarray(generator_bus, dtype=np.int64)
    xd1 = np.asarray(xd1_pu, dtype=np.float64)
    if gen_bus.shape != xd1.shape:
        raise ConfigurationError("one transient reactance per generator")
    if np.any(xd1 <= 0):
        raise ConfigurationError("transient reactance must be positive")
    n, g = y_bus.shape[0], gen_bus.size
    full = np.zeros((n + g, n + g), dtype=np.complex128)
    full[g:, g:] = y_bus
    if load_admittance is not None:
        full[g:, g:] += np.diag(np.asarray(load_admittance, dtype=np.complex128))
    yd = 1.0 / (1j * xd1)
    for k, (bus, y) in enumerate(zip(gen_bus, yd, strict=True)):
        full[k, k] += y
        full[g + bus, g + bus] += y
        full[k, g + bus] -= y
        full[g + bus, k] -= y
    return kron_reduce(full, np.arange(g))
