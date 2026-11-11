# v5 pre-flight checks (frozen model, 2026-11-09)

These are read-only checks on the frozen v5 model before transient-stability data generation. They ran on cairo-lab with at most 8 processes. Nothing in `~/data/grid/v5/frozen/` was written.

- Model: `frozen/model_v5_frozen.pkl`, sha256 `3393038e8946…`.
- Hour inputs: `frozen/hour_inputs_v5_frozen.pkl`, sha256 `c17d89d4…` (same as `hour_inputs_v5.pkl`).
- Defaults:
  - `build_connected_case(machine_data="entsoe_kundur")` (F1b), with HYGOV for hydro, min machine 50 MVA, and a CE equivalent.
  - `AndesEngine.shift_governor_limits=True`.
  - Per-TSO border split.
  - Hourly A78 outages OFF.
- Work folder on cairo-lab: `~/data/grid/v5/preflight/`. It holds the scripts, the logs, `dyn/<case>/*.pre.json`, `dyn_summary.csv`, the gate reports and `cct/`.

## 0. Code state

| item | result |
|---|---|
| HEAD | `6b5545f` (base) |
| tracked files vs `6b5545f + v5_freeze.patch` | all simulation code identical (`src/`, `scripts/`, `tests/`) |
| tracked files that differ | `apps/grid_map/app.js`, `apps/grid_map/qgis/grid_model.qgs`, `apps/grid_map/qgis/styles/lines_voltage.qml`, `apps/grid_map/style.css`, `docs/data_quality/README.md`: map app and docs only; the working tree is older than the patch (the patch adds e.g. the `osm_lines` layer) |
| untracked tarball contents | all `.py`/`.md` files present and identical; only macOS `._*` resource-fork files from the tarball are absent (irrelevant) |

**PASS for simulation.** The differences are in map/doc files, which no check here uses. Re-sync those files before the freeze commit.

## Summary

| # | check | result | key numbers |
|---|---|---|---|
| 0 | code = base + freeze patch | PASS (sim code) | 5 map/doc files differ |
| 1a | static, 9 named × tan 0.1/0.2, vs `s3393038e` | **PASS, identical** | PF hash `2f79b408…` = reference; sy/pp 18/18; islands 9; machines off 0; vmin 0.918; losses 23,080 MW (sy), 23,078 MW (pp); 10 overloaded rows; max loading 1.49 |
| 1b | static, 100 hours × tan 0.1/0.2 | **PASS, identical to `h3393038e`** | PF hash `6d4c6dfc…` = `h3393038e`; sy/pp 200/200; islands 9; pp NaN buses 8; machines off 0; vmin 0.918; losses 218,191 MW (sy), 218,086 MW (pp); 31 overloaded rows; max loading 1.49 |
| 2 | dynamic init, 109 cases (100 h at tan 0.1 + 9 named at tan 0.2) | **PASS** | init vs pypowsybl: dV ≤ 3.3e-5 pu, dA ≤ 0.014°, dP ≤ 0.99 MW, 0 lines out of tolerance; t = 0 residuals max\|f\| ≤ 3.3e-16, max\|g\| ≤ 9.8e-7; 0 governors and 0 exciters at a limit; 0 invalid H/D/x' |
| 2b | inertia vs ENTSO-E | REPORT (see below) | model German E_k = 0.25–0.95 (median 0.56) of the ENTSO-E RoCoF eq. 5.5–5.7 reference |
| 3 | flat 10 s, all 109 cases (incl. the 9 named and all 91 other hours) | **PASS** | drift 0.000° (raw and COI-relative), max\|ω−1\| = 0, COI f drift ≤ 4e-14 Hz, max ΔV ≤ 8e-9 pu, no oscillation |
| 4 | small-signal (EIG), 3 cases | PASS (no unstable), WARN (damping) | 0 eigenvalues with Re > 1e-6; poorest damping ζ = 0.7–1.0 % at 0.86–1.34 Hz |
| 5 | fault sanity, 2 cases × 5 lines | PASS with a remark | short (0.08 s): 10/10 stable; long: unstable on 3/5 lines per case; the median-x 380/220 kV lines are stable even at 1.0 s |
| 5b | CCT bracket vs v4.1 F1b (low_load_t10, lines 6 and 63, both terminals) | **PASS, identical** | 4/4 brackets equal F1b to 0.1 ms |
| 6 | timing | — | build + PF + init ≈ 6.5 s per case; flat 10 s ≈ 2.9 s; fault simulation (7 s window) median 3.9–4.8 s, max 15.5 s; CCT bracket ≈ 9 simulations, 77–105 s per line terminal |

**Go** for v5 case building and the CCT sweep. Two conditions apply (see "Failures and caveats").

## 1. Static (both solvers, as gate.py)

`gate.sh frozen/model_v5_frozen.pkl model_v5_5q.pkl prefix=v ref_prefix=s` → `preflight/gate_v3393038e_vs_s3393038e.md`: PASS, power-flow hash identical, 0 loading changes > 0.10.

`… prefix=w ref_prefix=u flags=hours=frozen/hour_inputs_v5_frozen.pkl` → `preflight/gate_w3393038e_vs_u3393038e.md`: formally PASS, but with a different hash.

- The `u3393038e` run had hourly A78 outages ON: 5,044 outage entries in its results. The frozen defaults have outages OFF. `u` is therefore the wrong reference for the defaults. It shows islands 16, machines off 48, vmin 0.863 and losses 244,830 MW.
- The frozen 100-hour run reproduces `h3393038e` (all 100 hours, outages OFF) bit for bit: hash `6d4c6dfc60a1376b`.

The 9 named cases have the same hash as the gate reference `s3393038e`.

## 2. Dynamic initialisation (v5 defaults)

Method: `preflight/v5pre.py` (v4.1 `mk.py` adapted to v5).

- Operating point: go4 `solve` for one tan φ, using `v5_pf_connected`, VDE 4120 Q(U), published compensation and the tap loop.
- Case: `build_connected_case` with the defaults.
- ANDES: PF, then the AndesEngine slack governor shift, then `TDS.init`.

| item | min | median | max |
|---|---|---|---|
| GENROU (German, ≥ 50 MVA) | 20 | 71 | 138 |
| GENCLS (CE equivalent) | 28 | 28 | 28 |
| TGOV1 / HYGOV | 44 / 1 | 90 / 7 | 148 / 31 |
| German H (s) | 2.7 | — | 4.3 |
| German x'd (pu) | 0.30 | 0.30 | 0.30 (Kundur) |
| German D | 0 | 0 | 0 (F1b) |
| neighbour H / D | 5.0 / 1.24 | | |
| ANDES vmin (pu) | 0.918 | 1.006 | 1.016 |

- Discrete limiters active at t = 0, over all models: none.
- `clamped_governors`: 0 in all 109 cases.

Kinetic energy, model (H·Sn of the German GENROU) against the ENTSO-E reference (Σ H_i·g_i/LF_i over the hour's A75 generation per TSO, RoCoF v17 Table 1), in GWs:

| hour | model DE | ENTSO-E DE | 50Hertz m/ref | Amprion m/ref | TenneT m/ref | TransnetBW m/ref | CE equiv. |
|---|---|---|---|---|---|---|---|
| high_load | 226.1 | 299.7 | 56.7 / 89.1 | 117.1 / 114.1 | 35.0 / 56.5 | 22.0 / 40.1 | 1498 |
| winter_peak | 162.4 | 280.4 | 59.2 / 96.3 | 66.8 / 102.1 | 22.0 / 48.8 | 18.8 / 33.2 | 1558 |
| summer_peak | 87.6 | 117.3 | 21.9 / 33.4 | 47.7 / 42.7 | 9.1 / 24.5 | 9.3 / 16.6 | 970 |
| baltic_import_low_load | 71.0 | 104.7 | 31.7 / 46.4 | 26.8 / 27.8 | 9.5 / 21.9 | 3.1 / 8.6 | 842 |
| median_load | 60.6 | 113.6 | 23.1 / 44.1 | 24.1 / 25.9 | 11.1 / 30.8 | 4.3 / 12.8 | 894 |
| very_light_night | 49.9 | 100.4 | 19.9 / 36.0 | 24.3 / 32.4 | 1.3 / 19.2 | 6.0 / 12.8 | 785 |
| high_wind | 34.9 | 88.5 | 10.9 / 28.4 | 17.5 / 22.5 | 4.2 / 29.1 | 2.5 / 8.5 | 1069 |
| high_solar | 28.7 | 73.0 | 16.0 / 28.6 | 9.3 / 18.5 | 1.8 / 19.3 | 1.6 / 6.7 | 715 |
| low_load | 26.2 | 69.7 | 14.6 / 27.5 | 8.7 / 14.5 | 0.8 / 18.8 | 2.0 / 8.9 | 787 |

Ratio model/reference over the 100 hours:

| | min | median | max |
|---|---|---|---|
| DE | 0.25 | 0.56 | 0.95 |
| 50Hertz | 0.27 | 0.59 | 0.80 |
| Amprion | 0.37 | 0.72 | 1.21 |
| TenneT | 0.00 | 0.27 | 0.92 |
| TransnetBW | 0.10 | 0.48 | 0.67 |

Machines at stations shared by two TSOs (about 2–3 % of E_k) are counted under neither TSO.

## 3. Flat run (10 s, no disturbance)

The pass thresholds are the same as v4.1 (`mk.py`):

- the run completes with t_end ≥ 9.9 s;
- maximum rotor-angle drift < 1.0°;
- max|ω − 1| < 1e-3 pu;
- max|ΔV| < 1e-2 pu;
- initial state vs PF: dV ≤ 1e-3 pu, dA ≤ 0.1°, line dP ≤ max(2 MW, 1 %).

109/109 cases pass, including the 9 named hours (tan 0.1 and 0.2) and all 91 other hours, which is more than the 5 random hours requested. Every case gives:

- rotor drift: 0.000° raw and COI-relative;
- max|ω − 1| = 0;
- COI frequency drift ≤ 4.4e-14 Hz;
- max|ΔV| ≤ 8.1e-9 pu;
- peak-to-peak COI-relative angle over 5–10 s: 0.

## 4. Small-signal (ANDES EIG, after PF)

| case | states | unstable | poorest-damped modes (f, ζ) |
|---|---|---|---|
| high_load_t10 | 1438 | 0 | 0.99 Hz 1.01 %, 0.93 Hz 1.06 %, 0.97 Hz 1.09 %, 0.86 Hz 1.15 % |
| low_load_t10 | 374 | 0 | 1.34 Hz 0.73 %, 1.32 Hz 0.74 %, 1.26 Hz 0.77 %, 1.16 Hz 0.83 % |
| very_light_night_t10 | 536 | 0 | 1.34 Hz 0.78 %, 1.32 Hz 0.78 %, 1.26 Hz 0.79 %, 1.16 Hz 0.84 % |

## 5. Fault sanity

Fault: 3-phase, 0.5 Ω, at the from terminal at t = 1 s. It is cleared by opening the line. The window is 7 s, and a run is unstable when a rotor angle is more than 180° from the COI.

Engine: `v2lib.Engine`, which is the repo AndesEngine with its defaults. Candidates: 988 lines in high_load_t10 and 992 in low_load_t10. In the table, S = stable and U = unstable (out-of-step onset time).

| case | line | kV | 0.08 s | 0.50 s | 1.00 s |
|---|---|---|---|---|---|
| high_load_t10 | Lechhausen–Meitingen (median x) | 380 | S | S | S |
| high_load_t10 | Eiberg–Y-Gelsenberg (median x) | 220 | S | S | S |
| high_load_t10 | Röhrsdorf–Hradec východ (border, CZ) | 380 | S | U (1.45 s) | — |
| high_load_t10 | Bärwalde–Schmölln (v4.1 line 6) | 380 | S | U (1.38 s) | — |
| high_load_t10 | Lauchstädt–Klostermansfeld (v4.1 line 63) | 380 | S | U (1.38 s) | — |
| low_load_t10 | Hamburg/Ost–Hamburg/Nord (median x) | 380 | S | S | S |
| low_load_t10 | Eiberg–Y-Gelsenberg (median x) | 220 | S | S | S |
| low_load_t10 | Röhrsdorf–Hradec východ (border) | 380 | S | S | U (2.07 s) |
| low_load_t10 | Bärwalde–Schmölln | 380 | S | U (1.39 s) | — |
| low_load_t10 | Lauchstädt–Klostermansfeld | 380 | S | U (1.37 s) | — |

None of the simulations failed numerically or needed a retry.

CCT bracket (`preflight/cct_bracket_v5.py` = `scripts/cct_bracket.py`, with only the xlsx path changed). Values are in seconds:

| low_load_t10 | v5 frozen | v4.1 F1b (e10) | v4.1 F1 | v4.1 original |
|---|---|---|---|---|
| line 6, from | 0.2503–0.2578 | 0.2503–0.2578 | 0.2429–0.2503 | 0.2801–0.2875 |
| line 6, to | 0.3691–0.3765 | 0.3691–0.3765 | 0.3617–0.3691 | 0.4136–0.4211 |
| line 63 (v5 index 62), from | 0.2652–0.2726 | 0.2652–0.2726 | 0.2503–0.2578 | 0.2875–0.3023 (failed) |
| line 63, to | 0.4507–0.4582 | 0.4507–0.4582 | 0.4359–0.4433 | 0.5027–0.5101 |

## 6. Timing (cairo-lab, 1 thread per process)

| step | time |
|---|---|
| PF, case build and TDS init | 5.6–8.9 s per case (median 6.5 s) |
| `TDS.init` | 0.05 s |
| 10 s flat run | 2.6–3.3 s |
| fault simulation (7 s window) | median 3.9–4.8 s; max 15.5 s (unstable, long window) |
| EIG | 1.2 s (high_load) |
| CCT bracket | 9 simulations per line terminal, 77–105 s |

At 8 processes, the full candidate set is about 990 lines × 2 terminals × ≈ 90 s / 8 ≈ 6.2 h per case. That gives about 112 h for the 18 named cases.

## Failures and caveats

1. **Hourly CE inertia exists only for the 9 named hours.** `neighbour_inertia_entsoe.csv` has rows only for those 9 hours. For the other 91 hours, `continental_europe_equivalent(..., "entsoe_hourly")` raises `DataError`, so this pre-flight used `inertia="central"` for them. **Before the 100-hour v5 case building**, either fetch A75 for the 91 hours into the table or fix "central" in FREEZE.md as the documented default for them. This blocks the 100-hour case building with the stated defaults, but not the named-case CCT sweep.
2. **German inertia is 25–95 % (median 56 %) of the ENTSO-E reference**, and TenneT is lowest (median 27 %, 0 at times). Likely causes:
   - the builder keeps only machines ≥ 50 MVA as GENROU, so smaller units are static;
   - the reference counts all synchronous generation per type at S/LF, including biomass, waste and run-of-river, which the model aggregates or treats as static;
   - TenneT's thermal fleet is small in most hours.

   This is the same modelling choice as v4.1 F1b, so it is consistent with the e10 labels. Note it in KNOWN_LIMITS; it does not block.
3. **Small-signal damping is 0.7–1.1 %** at 0.9–1.3 Hz. This follows from D = 0 (F1b) and no PSS. There are no unstable modes and the flat runs show no growth. Long post-fault oscillations decay slowly, so stability windows of 5–6 s after clearing can still be swinging. It is the same as F1b and does not block.
4. **Long clearing is not unstable on every line.** The median-x 380 kV and 220 kV lines are stable even at 1.0 s, because they are far from large machines (as the "above" brackets in v4.1 were). The lines near machines and the border line behave as expected. In the sweep, many lines will come out as "above" (CCT > 1.0 s).
5. **The CCT is identical to F1b** on the 4 brackets. The v5 changes do not touch the Lusatia–Central Germany area at low_load, and the v5 build path reproduces the F1b numbers. This is a consistency check, not new information on the v5 changes.
6. **Map/doc files on cairo-lab differ from the freeze patch.** Re-sync them before the freeze commit. Simulation is not affected.

## Go / no-go

- **v5 case building, 9 named hours (t10/t20): GO.** Init, flat run, EIG and faults are all clean.
- **v5 case building, all 100 hours: GO with condition 1.** Decide the CE inertia source for the 91 non-named hours first.
- **CCT sweep: GO.** The brackets reproduce F1b, and with the defaults there are no numerical failures and no clamped governors. Budget about 6 h per case at 8 processes.
