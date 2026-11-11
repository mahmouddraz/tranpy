# v5.1 freeze (owner decision 2026-11-09)

v5.1 replaces the v5 freeze for the transient-stability data generation. The v5 freeze in `~/data/grid/v5/frozen/` on cairo-lab is kept unchanged. All work for this freeze is in `~/data/grid/v5/frozen_v5_1/` (read-only) and `~/data/grid/v5/preflight_v5_1/`.

## What changed vs v5

There is one model change: issue 14 #1 (A016, candidate c1). It is CHANGELOG entry 45.

- **The change.** The two circuits osm-14135-0 and osm-14135-1, Sechtem–Brauweiler 380 kV, have status `in_operation` → `duplicate` (status_date 2026-11-09). They no longer take part in the power flow.
- **Evidence** (details in `docs/data_quality/issues/14-remaining-halos/model_check.md`):
  - JAO Core SGM r9 (published 2026-03-16) has no Sechtem–Brauweiler element.
  - Sechtem has 9 Amprion 380 kV rows. Brauweiler has only BRAUWL W (to Rommerskirchen) and BURBAC O (to Knapsack).
  - There is no such element in r1, and no change-log row adds or removes one.
  - Both export circuits come from OSM way 24431541 (ref 4511). That way is the tower line of VILLE O/W, and the route overlaps VILLE O/W by 95 %.
- **Independent diff of the model** (`preflight_v5_1/diff_models.py`, `frozen/model_v5_frozen.pkl` vs `step5/cand_c1.pkl`):
  - The only differences are `status`, `status_date` and `status_evidence` of osm-14135-0/1, in both `grid_de_circuits` and the CIM `ac_line_segment`.
  - The 6 matching rows are appended to `grid_de_field_changes`. The first 708 rows are identical.
- Code, hour inputs and defaults are the same as v5.
- Issue 14 proposals #2–#9 are kept as in v5. They are recorded in KNOWN_LIMITS.

## Freeze contents (`~/data/grid/v5/frozen_v5_1/`, read-only)

| file | sha256 |
|---|---|
| `model_v5_1_frozen.pkl` (copy of `step5/cand_c1.pkl`) | `cc4589c173f9466408d44b043105b5213c12fed17bc9ce2ad2f9a4cc922b723e` |
| `model_v5_1.changes.csv` (copy of `cand_c1.pkl.changes.csv`) | `bb21167cd37cca78e48f77b01d7c4da64b9fb3bc2ac25969e0ee3847a2c7b789` |
| `hour_inputs_v5_frozen.pkl` (copy of `frozen/`, unchanged) | `c17d89d49d53baa6bf6218d38771ad5c424a3e5af50f3222265dd55597529cf0` |
| `v5_1_freeze.patch` (`git diff HEAD` on base `6b5545f`, Mac repo) | `944360fcc74d061046281e4264a8281fcc0fb7e989046bf35ae41361902e1eaa` |
| `v5_1_freeze_untracked.tgz` (untracked src/scripts/tests/data, no xattrs) | `5cbc66e88cd996eb5dfd79b816848667f38a1ecda5f7894321ba44aa4fb1143f` |
| `FREEZE.md` | `402b11cc1ff44d8dafb718e7a3202d655dfc91084214b0365efda243f3edc581` |

Code snapshot vs the cairo-lab working tree:

- The simulation code is identical.
- Four map-only files differ: `grid_gis.py`, `grid_network.py`, `scripts/map_acceptance.py` and `scripts/map_edge_review.py`. The snapshot holds the Mac versions. No check here uses these files.

Defaults (FREEZE.md):

- F1b machine data `entsoe_kundur`;
- `shift_governor_limits=True`;
- per-TSO border split, with missing hours → capacity split;
- hourly A78 outages OFF;
- neighbour inertia: `entsoe_hourly` for the 9 named hours and `central` for the other 91 hours (**owner decision**).

New gate references:

- `acc4589c1`: 9 named cases, power-flow hash `7d48c2390f220851`;
- `bcc4589c1`: 100 hours, power-flow hash `34910793a90df9ac`.

## Gate (hard checks)

| run | reference | sy conv | pp conv | islands | pp NaN buses | machines off | store/CIM sync | result |
|---|---|---|---|---|---|---|---|---|
| 9 named × tan 0.1/0.2 (`acc4589c1`) | `s3393038e` | 18/18 | 18/18 | 9 = 9 | 8 = 8 | 0 | 0 | **PASS** |
| 100 hours × tan 0.1/0.2 (`bcc4589c1`) | `h3393038e` | 200/200 | 200/200 | 9 = 9 | 8 = 8 | 0 | 0 | **PASS** |

These checks are reported and do not block:

- **Losses:** +18/+19 MW (sy/pp) over the 9 cases and +218/+217 MW over the 100 hours.
- **Overloaded rows:** 10 = 10 and 31 = 31.
- **Minimum 220/380 kV voltage and maximum loading:** 0.918 and 1.49, both unchanged.
- **Loading changes > 0.10:** only on BRAUWL W, WABERG W and BURBAC O, plus the removed osm-14135 lines.
- **Named-case hash:** equals the step5 `scc4589c1` run.

## Pre-flight (thresholds as in `docs/v5_preflight.md`)

| # | check | result | key numbers |
|---|---|---|---|
| 1 | dynamic init, 109 cases (100 h at tan 0.1, 9 named at tan 0.2) | **PASS** 109/109 | dV ≤ 3.3e-5 pu, dA ≤ 0.014°, dP ≤ 1.0 MW, 0 lines out of tolerance; t = 0 max\|f\| ≤ 3.3e-16, max\|g\| ≤ 9.9e-7; 0 governors clamped, 0 limiters at t = 0; 0 invalid/implausible parameters; CE inertia `entsoe_hourly` 18 cases, `central` 91 cases |
| 1b | inertia vs ENTSO-E (report) | REPORT | German E_k / reference 0.25 / 0.56 / 0.95 (min / median / max), the same as v5 |
| 2 | flat 10 s: 9 named × tan 0.1/0.2 + 5 random hours (seed 51: 20251226T11, 20260210T16, 20260228T06, 20260703T10, 20260805T09) | **PASS** 23/23 | rotor drift 0.000° (raw and COI), max\|ω−1\| = 0, COI f drift ≤ 1.1e-14 Hz, max\|ΔV\| ≤ 7.6e-9 pu, no oscillation |
| 3 | fault sanity, high_load_t10 + low_load_t10 × the v5 lines (same bus pairs: 380 kV median-x, 220 kV median-x, border Röhrsdorf–Hradec, v4.1 lines 6 and 63) | **PASS** | 25/25 simulations have the same label and out-of-step time as v5; 0 numerical failures |
| 4 | CCT brackets, high_load_t10, BRAUWL W / VILLE O / WABERG W, both terminals | **PASS** | 6/6 brackets identical to the step5 combined result (required: within one bracket width, 7.5 ms) |

CCT detail for high_load_t10, in seconds. Position 0 is the from terminal and position 1 the to terminal.

| line | pos | v5 frozen | step5 combined | v5.1 | v5.1 − v5 |
|---|---|---|---|---|---|
| BRAUWL W Brauweiler–Rommerskirchen (178) | 0 | 0.406–0.414 | 0.473–0.481 | 0.473–0.481 | **+67 ms** |
| BRAUWL W | 1 | 0.339–0.347 | 0.339–0.347 | 0.339–0.347 | 0 |
| VILLE O Rommerskirchen–Sechtem (448) | 0 | 0.339–0.347 | 0.339–0.347 | 0.339–0.347 | 0 |
| VILLE O | 1 | 0.369–0.377 | 0.399–0.406 | 0.399–0.406 | **+30 ms** |
| WABERG W Knapsack–Sechtem (453) | 0 | 0.317–0.325 | 0.317–0.325 | 0.317–0.325 | 0 |
| WABERG W | 1 | 0.377–0.384 | 0.406–0.414 | 0.406–0.414 | **+30 ms** |

In step5, low_load_t10 shifted by +52, +22 and +15 ms on the same terminals. The +15…+67 ms shift is therefore due to c1 alone; the other step5 candidates do not contribute.

## Known limits added (KNOWN_LIMITS.md, "v5.1 freeze", backup `.bak45`)

- **L45a**: Issue 14 proposals #2–#9 are kept. Their circuit counts rest on OSM only:
  - Bixterheide/Garenfeld osm-5010;
  - Remptendorf osm-4706;
  - Niederaußem osm-4291;
  - Schwelgern osm-5127;
  - Schwelgern/Beeck osm-4386;
  - LMG osm-4306;
  - Nordleitung/Südleitung osm-4329.
- **L45b**: Zwönitz 577/578. The topology and the length are unpublished. The model uses the curated 30.6 km; OSM gives 24.5 km.
- **L45c**: German inertia is low against the ENTSO-E reference. The median ratio is 0.56, and TenneT's is 0.27. Only machines ≥ 50 MVA are modelled dynamically.
- **L45d**: Damping is weak, about 1 %, because D = 0 and there is no PSS.
- **L45e**: Neighbour inertia is `central` for 91 of the 100 hours.

## Go / no-go

**GO for the 9-case CCT sweep on v5.1.**

- Every hard gate passes.
- Initialisation and flat runs are clean.
- The fault sanity results are the same as v5.
- The Sechtem/Brauweiler CCT brackets reproduce step5 exactly.

The budget is unchanged from v5, at about 6 h per case with 8 processes. The limits L45a–e apply. L45e affects only the 100-hour case building, not the 9 named cases.
