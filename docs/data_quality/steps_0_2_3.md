# PLAN steps 0, 2, 3 (2026-11-08)

All work is on cairo-lab in `~/data/grid/v5/`. Nothing in the power flow changed. `model_v5.pkl` (ec0ba792…) and `model_v4final.pkl` were not modified.

## Step 0: automatic check and change log
- `gate.py` and the wrapper `gate.sh MODEL [REF=model_v5.pkl]` run `run_check.sh` (go4 build and solve, 9 cases × tan φ 0.1/0.2, pypowsybl and pandapower) under the tag `g<sha8>`. A complete earlier run with the same tag is reused.
- Hard gates (the model fails if it is worse than the reference):
  - convergence 18/18 in both solvers;
  - islands (topological components in the solved case);
  - pandapower NaN buses;
  - in-service machines outside the main island;
  - overloaded rows;
  - lowest 220/380 kV voltage in each solver;
  - total losses in each solver (tolerance `loss_tol_mw`, default 0);
  - store/CIM line-value mismatches (added in step 2).
- The report lists every line whose loading changes by more than 0.10 in either solver, with its name. Lines that are new or removed are listed too.
- Each run writes `gates/gate_<sha>_vs_<sha>.{md,json}`. The output includes a power-flow result hash, a sha256 over every flows pickle (both solvers' loadings, bus voltages, island counts) and over the results table.
- Determinism: a fresh run of model_v5 reproduces the hash of the original v5 run, `e23b3c00686285f6`.
- `CHANGELOG.md` starts with the three v5 changes (C-Bürs, 328, DUEMM). These were not gated one by one because the gate did not exist yet. Entry 4 is step 2.

## Step 2: one source of the line values (D1/D2)
- The authoritative copy is the CIM `ac_line_segment`, which the power flow reads. The store `grid_de_circuits` length/R/X/B is now derived from it (`b_us = bch_s·1e6`) by `line_values.py make`.
- `line_values.py check MODEL` raises `LineValuesOutOfSync` (exit 1) on any mismatch. The gate fails a model that has more mismatches than its reference.
- The two HVDC store rows have no AC segment and are left out.
- At tolerance 1e-9 the actual difference was **127 rows / 322 values**: x 99, b 98, r 97, length 28. That is more than the 100 rows stated in DQ04, and 126 of them differ by more than 0.1 %. All are synced, including all 29 group F rows.
- The 11 doubled tie-line rows (group C) are relabelled `data_quality` published → estimated in both the store and the CIM. Values are unchanged.
- Every field change is logged in `grid_de_field_changes`, plus `model_v5_1.pkl.sync_diff.csv` and `model_v5_1.pkl.changes.csv`.
- Ratings were not touched: imax_a differs on 75 rows, and go4 reads the store's `rating_basis`/`dlr_*` columns.
- Result: `model_v5_1.pkl` sha 9e2fd914… (`model_v5_1.sha256`). Gate vs v5: **PASS, power-flow hash identical (e23b3c00686285f6), 0 lines > 0.10**.

## Step 3: station table and checks (detection only)
`stations/stations.py model_v5_1.pkl stations` writes the following to `~/data/grid/v5/stations/`:
- tables: `stations`, `voltage_levels`, `bus_nodes`, `junctions`, `transformers` (parquet);
- the layers stations, voltage_levels, bus_nodes and junctions in `stations.gpkg` (EPSG:4326);
- `checks.csv`.

What the table holds:
- 455 stations, 419 of them with an OSM polygon footprint; 539 voltage levels and bus nodes; 94 junctions with positions, legs and tee_positions; 144 CIM transformers.
- Polygon matching follows class 05, in this order: the osm_bus reference, then the dq05 name match, then the nearest HV polygon, all within 500 m. This matches 496/508 non-junction nodes. dq05 matched 477/509 by name only.

Checks (rows in checks.csv):
- junction dead end (fewer than 3 legs): 17 (error)
- isolated node: 4 (emden/ost, gutersloh, hesseln 220 kV, osm/1463882574)
- dead-end node: 65. 14 are errors (no 110 kV transformer, no injection); the rest are radial feeders.
- shared coordinate: 32
- level without a transformer to the other level: 32 (16 stations)
- node in two stations: 59
- node in no station: 12 (+29 foreign, info)
- junction inside or within 50 m of a station: 16; station node within 300 m of a junction: 27
- possible misspelling: 9; duplicate name: 3
- junction split across voltage levels: 1
- junction unlocated: 1
- foreign label inside Germany: 2

Several check types include intended cases, so they need review before any fix:
- radial plant connections among the dead-end nodes;
- Meitingen/Metzingen/Meiningen, which are real distinct stations;
- "50h" or "c1/c2" split-owner nodes.

PLAN A1–A4 are all detected:
- **A1**:
  - `herbentingen` / `herbertingen` misspelling;
  - `herbentingen|380` dead end, labelled foreign although in Germany, at the same point as `y-hoheneck#herbtg w`, inside the Hoheneck polygon;
  - `y-herbertingen#herbtg w` and `y-hoheneck#herbtg w` are junction dead ends inside the stations.
  - The X 44–61× published is a parameter issue. It is not a structure check and is not detected here.
- **A2**:
  - `y-erzhausen#4` is split across 220/380 kV;
  - `y-erzhausen#2|220` and `#4|220` are dead ends;
  - `#4|380` and `#5|380` share a point.
- **A3**: `y-kuhmoos#bl` and `y-laufenburg#rtge` are dead-end junctions at an identical point.
- **A4**: `osm-4706` is unlocated, inside the Remptendorf polygon (23 m from a station node), and its six legs are all 13.35 km.
