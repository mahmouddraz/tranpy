# v5 operating hours: dispatch inputs (2026-11-08)

Status: inputs only. No case built, nothing simulated. v4.1 frozen inputs untouched.

## Year and reason

Window **1 Nov 2025 - 31 Oct 2026** (the `data2026` window), not calendar 2025.

* The v4.1 hours are **not 2021**. They already come from this window (`pf_v4final/input_v4g_*.pkl`,
  `go4.py one(("v4g","2026",...))`): baltic_import_low_load 2025-11-03 01:00, high_load 2026-01-29 10:00,
  winter_peak 2026-01-22 07:00, median_load 2026-04-30 16:00, summer_peak 2026-06-29 08:00,
  high_solar 2026-08-13 11:00, very_light_night 2026-08-23 01:00, low_load 2026-08-30 01:00,
  high_wind 2026-10-23 18:00 (UTC). 2021 is used only for the real-event validation (8 Jan / 17 May 2021).
* The window is the most recent complete 12 months and the closest to the 2026 grid (Baltic states in CE
  since 9 Feb 2025; Baltic Cable outages are real, see `dispatch2026/check_notes.md`).
* Coverage: 8,760 hours, SMARD load per TSO area and ENTSO-E generation per control area with no gap;
  cross-border flows from ENTSO-E with Energy-Charts fill as in `data2026.py`. Data were already cached
  and checked (Destatis/ISE comparison in `check_notes.md`).
* The sampling plan keeps the 9 v4.1 extremes inside v5; same window means they stay consistent.
  Calendar 2025 (`pf_final2026/data2025.py`) is also cached and can be swapped in by rerunning `select.py`
  on it.

## Selection (work/v5_hours/select.py, seed 20261108)

Features per hour: DE load (SMARD), wind (B18+B19) and solar (B16) from ENTSO-E per control area,
net import = -sum of the border exchanges X, synchronous share = (generation without B16/B18/B19)/load,
RES share = (B01,B09,B11,B12,B15,B16,B18,B19)/generation.

Cells: load tercile x wind tercile x solar day (B16 >= 1 GW) / night x import sign = 36; 34 non-empty
(no mid/high-load night hours with low wind and net export).
Ranks 1-9: the v4.1 hours. Then rounds; per round every non-empty cell, in seeded random order, gives one
hour: the season least represented so far, then the 6-h band of the day least represented, then random;
at most one hour per calendar day. Ranks are nested: option (a) = ranks 1-60, option (b) = ranks 1-100.
tan phi 0.1/0.2 drawn per hour from the same generator (column `tan_phi`).
Result: 100 hours, 25 per season, 25 per 6-h band, 1-5 per cell (median 3).

## Data fetched

Only new download: ENTSO-E A75 (processType A16) for the 31 CE countries + DE at the 91 new hours
(3,200 small files incl. the 9 old hours, in `entsoe_2026/ce_a75`, ~1 MB), the input of the
CE-equivalent inertia. Everything else reused from cache (ENTSO-E A75/A73 per German control area and
A11 flows, SMARD load/series, Energy-Charts physical flows for SE4/NO2/LU). FCR is constant
(564 MW German, FROZEN_DYN_v41), so nothing to fetch. ENTSO-E data: research use only.
No A75 at all for TR and UA (as in v4.1); AL (60 hours), MD (11), MK (1) filled with the mean of the
country's other v5 hours (compute.py rule).

## Outputs (work/v5_hours/, mirrored to cairo-lab:~/data/v5_hours/ with raw/ inputs)

| file | content |
|---|---|
| selected_hours.csv | rank, name, time, cell, season, band, tan phi, features |
| data_v5.py | drop-in for `data2026` (A,G,PU,nat,L,X,SRC + v5 `hours`, `tan_phi`, `option_a`); go4 `one` / mk.py unchanged when `sys.modules["data2026"]` points to it |
| hour_inputs_v5.pkl | per hour the exact objects go4 `one` passes to `dispatch_hour` (g, L, X, PU, SRC) |
| neighbour_inertia_entsoe_v5.csv | same format as `src/tranpy/simulation/neighbour_inertia_entsoe.csv` (package CSV not changed) |
| summary_hours.csv | load, wind, solar, RES share, sync share, net import, E_neigh / E_DE / E_CE (GWs) |
| validation_hours.csv, validation_fuel.csv, inertia_missing.csv | checks below |

## Validation

* No missing hours: 0 NaN in load, generation or flows at the 100 hours; one hour without published
  LU exchange (set 0 as in data2026).
* Load: sum of the four TSO areas vs SMARD DE: max deviation 0.001 %. (The case load is SMARD itself.)
* Generation by fuel: control-area sum vs national ENTSO-E A75 and vs SMARD: identical for lignite, gas,
  onshore, offshore, solar (< 10 MW mean); total 1.3 % lower (max 3.3 %), mainly other_conv (-0.5 GW mean,
  waste/other not split by control area), biomass (-90 MW), other_ren (-53 MW). This gap is what go4's
  `with_unreported_feed_in` already closes, as in v4.1.

## Comparison with the 9 v4.1 hours

| | v4.1 (9) | new (91) |
|---|---|---|
| load GW (min/median/max) | 32.6 / 55.4 / 78.2 | 37.8 / 52.7 / 69.9 |
| wind GW | 3.6 / 21.0 / 53.4 | 1.6 / 16.6 / 39.6 |
| solar GW | 0 / 1.4 / 58.1 | 0 / 2.2 / 46.4 |
| RES share | 0.40 / 0.70 / 0.89 | 0.22 / 0.60 / 0.93 |
| net import GW | -12.9 / 0.4 / 11.0 | -14.9 / 0.1 / 11.9 |
| sync share | 0.23 / 0.36 / 0.62 | 0.18 / 0.48 / 0.79 |
| E_CE GWs | 793 / 1013 / 1840 | 704 / 1183 / 1804 |

The v4.1 hours are the extremes; the new hours fill the interior and add low-RES, high-sync and deeper
export/import points.

## Not done

Case building, power flow feasibility rejection (plan sec. 3), simulation, commits.
