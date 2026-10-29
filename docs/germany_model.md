# German transmission grid model

`AndesEngine(system_name="germany")` simulates the German 220/380 kV transmission grid,
built from open data. The case file is bundled (`src/tranpy/resources/cases/germany.xlsx`)
and can be rebuilt with `tranpy.simulation.cases.build_germany_case`.

```python
from tranpy.simulation import generate_events
from tranpy.simulation.andes import AndesEngine
from tranpy.simulation.protocol import StressTestSampler

engine = AndesEngine("germany", name_width=4, n_jobs=8)
sampler = StressTestSampler(n_lines=engine.n_lines, n_loads=engine.n_loads)
engine.simulation_time_s = sampler.simulation_time_s
events = generate_events(engine, sampler.sample(300, seed=0),
                         criterion="angle_360", protocol="stress_test")
```

## Source

PanTaGruEl v0.1, a model of the continental European transmission grid
([Zenodo 2642175](https://doi.org/10.5281/zenodo.2642175), CC BY 4.0). Please cite when
publishing results:

- L. Pagnier, P. Jacquod, *Inertia location and slow network modes determine disturbance
  propagation in large-scale power grids*, PLOS ONE 14(3): e0213550, 2019.
- M. Tyloo, L. Pagnier, P. Jacquod, *The key player problem in complex oscillator networks
  and electric power grids*, Science Advances 5(11): eaaw8359, 2019.

The archive is downloaded and verified by checksum (`tranpy.simulation.cases.PANTAGRUEL`).

## The model

| | |
|---|---|
| Buses | 553 (342 at 380 kV, 209 at 220 kV, 2 at 132 kV) |
| Lines | 1319, of which 1224 transmission lines (fault candidates) and 95 transformers |
| Loads | 551, 68.2 GW in total |
| Generators | 86 synchronous units, 104.6 GW (lignite 31, gas 14, pumped storage 13, hard coal 11, nuclear 7, other 10) |
| Frequency | 50 Hz |
| Inertia constant H | 3.0-8.7 s (median 6.0 s) |

## How it is derived

1. Buses with country `DE`; lines to neighbouring countries are removed, so Germany runs
   as an island. The largest connected part is kept (4 buses drop out).
2. Line impedances as given (per unit on 100 MVA). Branches between 220 and 380 kV buses
   are transformers.
3. Loads from the bus demand at unity power factor (the source has no reactive demand).
4. All generators run at the same share of their capacity (about 66 %), covering demand
   plus 2 % losses. With merit-order dispatch, generation concentrates in a few regions
   and the AC power flow does not converge without the neighbours' support.
5. Generators are classical machines (ANDES `GENCLS`: swing equation behind a transient
   reactance). Inertia and damping are converted from the source's values on the system
   base: H = m · 2π f0 · Sb / (2 Sn), D = d · 2π f0 · Sb / Sn. The transient reactance
   (0.3 p.u. on the machine base) is an assumption.

The AC power flow converges with bus voltages of 0.964-1.02 p.u.

## Limitations

The model is suitable for studying rotor-angle (transient) stability on a realistic
topology. It is not a validated model of the German system:

- **No controllers**: classical machines have no excitation system, governor or power
  system stabiliser. Voltage dynamics and post-fault frequency recovery are not represented.
- **Isolated from Europe**: without the interconnection, inertia and short-circuit
  strength are lower than in reality; instability is more likely than in the real grid.
- **Generation fleet of the source (about 2016-2018)**: nuclear units are still in service;
  wind and solar are not modelled (their demand share is covered by the synchronous units).
- **Simplified network**: no line charging, no transformer taps, no generator step-up
  transformers; a few long radial connections (a generator behind a single line) become
  separate islands when that line opens and are then not simulated.
- **Fixed dispatch**: one dispatch pattern scaled with the load; the stress-test protocol
  varies loading but not the generation mix.

## First results

Stress-test protocol (fault at 1 s, clearing 0.05-0.40 s), 32 scenarios: all simulations
completed, 9 % unstable, about 2 s per scenario on 8 processes.

## Operator grid data (Core Static Grid Model)

For current topology and real line parameters, `tranpy.grid_data.load_core_grid_model()`
reads the [Core Static Grid Model](https://www.jao.eu/static-grid-model) that the Core
TSOs publish through JAO every six months (release 9, status March 2026; downloaded and
verified by checksum). The German part:

| Operator | Lines | Tie-lines | Transformers |
|---|---|---|---|
| Amprion | 350 | 51 | 41 |
| TenneT DE | 260 | 51 | 48 |
| 50Hertz | 142 | 19 | 32 |
| TransnetBW | 96 | 32 | 12 |

Each line has its operator, substations, voltage, R, X, B (ohm, microsiemens), length,
current rating and EIC code; each transformer its parameters and phase-shifter data.
Line routes and transformer locations come from the map published with the model.
"Tie-lines" include lines between two German operators.

```python
from tranpy.grid_data import load_core_grid_model
from tranpy.maps import plot_operator_map

model = load_core_grid_model()
plot_operator_map(model)
```

Not in the source: generators, loads, spur and generator connection lines, and dynamic
data. 389 German lines (Amprion, TenneT, TransnetBW) have no static current rating. The
map draws most lines as straight segments between substations (topology, not exact
routes). The data is not in CIM/CGMES format; the operators' CGMES models are not public.
