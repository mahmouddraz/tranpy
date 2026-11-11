# v5 model changes (basis: 2026 grid, JAO Core SGM 9th release)

Owner decision (2026-11-08): v5 keeps the curated 2026 grid of v4final as its basis (JAO 9th release, 2026-03). Operating hours move to a recent year in a separate step. The grid is not rolled back to 2021.
Result: `cairo-lab:~/data/grid/v5/model_v5.pkl`, sha256 `ec0ba79279e9f3311c92f6129fadf632b0d0a6f544494d17871412e26665e8b2`. The summary is in `MODEL_v5.md` and the file hashes in `model_v5.sha256`.
Build: `apply_v5.py model_v5.pkl burs,p328,duemm`, applied to `model_v4final.pkl` (read only, sha256 b1a98e23…8ddc16, unchanged). Every field change is logged in `grid_de_field_changes`.

## Validation setup
- `run_check.sh TAG MODEL` runs `go4.py` on cores 16–31 (cairo-lab), with the same settings as pf_v4final: 9 cases × tan phi 0.1/0.2, pypowsybl and pandapower. `summ.py TAG REF` summarises a run and compares the per-line loadings with REF.
- pandapower **3.5.6** is installed in `~/code/tranpy/.venv`. It pins pandas~=2.3, and the install downgraded pandas, so pandas 3.1.0 and numpy 2.5.4 were restored, because the model pickles need pandas 3.
  `pf_v5/pp_compat.py` patches pandapower's `_set_entries` (row enlargement on bool columns fails under pandas 3). On case118 and on a 3-bus test it gives the same results as unpatched pandapower under pandas 2.3.3.
- Reference: v4final run `ref`. All 18 runs converge in both solvers, EHV 0.914–1.071 pu, 110 kV 1.013–1.025 pu, max loading 1.48, 7 overloads, solver loading agreement ≤ 0.022, bus mismatch ≤ 0.919 MW, 10 buses unsolved in pandapower. This matches FROZEN_MODEL_v4final.
- Each change was checked against the step before it. Accepted only if: all 18 runs converge in both solvers; the voltage range is no worse; no new overload; solver agreement ≤ 0.022 and bus mismatch ≤ 1 MW; no new islands; and the stated improvement criterion is met.
- Grid tests (pytest, cairo-lab): 718 passed, 1 skipped. The pandapower tests pass with the shim; without it, 5 fail on the pandas 3 incompatibility.

## Changes
| id | asset | evidence | applied change | improvement criterion | validation vs previous step | verdict |
|---|---|---|---|---|---|---|
| C-Bürs | BLUDNZ W Y-Buers 11T-D4-D7-00001W, DELLM O Y-Buers 11T0-0000-0954-P (tees at Werben, option A) | bursa_corridor_findings: OSM route Bürs–Werben 47.0 km. The far legs match a tee at Werben within 3 % / 0.1 %. Published X 9.08/9.13 Ω means 0.19 Ω/km over 47 km, which is not physical for a 220 kV OHL | Length 47.0 km. BLUDNZ W: R 2.568 / X 18.847 Ω / B 128.8 µS (per km of legs 1620-L + 0962-R, X 0.401 Ω/km). DELLM O: R 2.399 / X 15.494 Ω / B 156.4 µS (per km of leg 1374-4, X 0.330 Ω/km). data_quality = corrected | X/km of the Bürs legs in the 0.30–0.42 Ω/km range of the same circuits | 18/18 in both solvers. Voltages, max loading, overloads, solver agreement 0.022, mismatch 0.919 MW and islands identical to ref. Largest loading change 0.013 (on the BLUDNZ legs themselves) | **keep** |
| C-2026 (328) | Parchim/Süd–Güstrow 328 = osm-5006-1 | JAO 9th release change log 2026-03 (50HERTZ): "Parchim/Süd - Güstrow 328, 11TD8L328------I, Removed Element". 50Hertz is replacing the 220 kV line with the 380 kV Ersatzneubau Güstrow–Parchim/Süd (plan approval 12/2023; the old line is dismantled before the new build) | status decommissioned, status_date 2026-03-01. Circuit 321 (osm-5006-0) kept | The model matches the TSO's 2026 data submission | 18/18 in both solvers. Global metrics identical, mismatch 0.920 MW, islands unchanged. 321 goes from 0.20 to 0.36 at most (high_solar). 332-322 up by ≤ 0.04 | **keep** |
| C-dup | Ohlensehlen–Wehrendorf DUEMM S1 11T0-0000-0978-5 (Amprion) and DS1 11T-D2-D7-00004B (TenneT) | JAO change log 2024-03-04 (Amprion): "Ohlensehlen - Wehrendorf DUEMM S1, EIC corrected" to 11T-D2-D7-00004B. 2024-02-29 (TenneT): DS1 00004B "New line. Was previously a tripod line" (1115-Z). The r9 tie-line sheet keeps both rows (X 8.02 / 8.006 Ω). JAO r1 has one circuit | DUEMM S1 set to status duplicate. DS1 kept with its current parameters | One circuit in the model, as published | 18/18 in both solvers. Voltages, max loading, overloads, agreement 0.022 unchanged. Mismatch 0.982 MW (≤ 1). Islands unchanged. DS1 goes from ≤ 0.25 to ≤ 0.39; the neighbouring lines change by ≤ 0.13 | **keep** |
| C-Merzen | Hanekenfaehr–Merzen FREREN N, 11T0-0000-2480-1 | JAO notice 23/09/2025: in service from 26/09/2025, Imax 2720 A. The notice gives no R/X/B or length, and FREREN N is not in the 9th release (2026-03): Merzen only has BRAGAU O to Westerkappeln. No TSO parameter source was found | none (no invented parameters) | — | — | **gap recorded, stub kept** |
| C-params | Library estimates that have JAO values. Parchim/Süd–Güstrow 328 (osm-5006: library X 14.44 Ω vs JAO r1 X 57.56 Ω) | The 9th release has no 328 row (removed, see C-2026). The r1 value (R 15.50 / X 57.56 Ω over 52.3 km = 1.10 Ω/km) is not physical for a single 220 kV OHL, and the two osm-5006 rows cannot be told apart. The other 37 estimated circuits with r9 values: none has a JAO length within ±10 % of the model length together with a plausible X/km (ratio 0.005–3.2). The v4 curation (grid_de_library_changes) replaced those values on purpose because they cover a partial length (often the German share of a tie line) or are placeholders | none | Published value applied only if the circuit identity is certain | — | **not applied** |
| C-2026 (other) | Model vs r9 | All 959 JAO-EIC circuits of the model are in r9. Of the r9 removals (releases 8–9), only 328 matches one of the model's 61 OSM/curated additions. Klixbüll/S–Endrup: Energinet puts it in service on 10 Mar 2026 (DK Havenergi, 9 Mar 2026), so it is kept. Perleberg 516: r9 lists it at 380 kV, so it is kept. Merzen station "full commissioning 2028" (Amprion), but r9 and the JAO notice show it switched in | none | — | — | **no further change** |

Duplicate scan (whole model, in-operation circuits). Bus pairs served by more than one TSO: only DUEMM S1/DS1 confirmed. The others are not duplicates:
- Brunsbüttel 949/951/952: different circuits.
- Garenfeld BIGGE W leg vs osm-5010-0 "Station Halfeshof" (ENERVIE): a different owner, not confirmed.

Shared EICs (12 rows) are tee legs of one circuit (547/548 Eisenhüttenstadt, 302 Wuhlheide, WALSUM, HERBTG W 00016J). OSM/curated rows on shared pairs are OSM-tagged parallel circuits.
Röhrsdorf–Zwönitz 577/578 (map "4×" near Zwönitz, way 157332865) are **not** duplicates: osm-14312-0/1 run Markersbach–Zwönitz (21.7 km) and curated 577/578 run Zwönitz–Röhrsdorf (30.6 km), two series sections meeting at Zwönitz. r9 has no 577/578 rows.

## Overall vs v4final
18/18 runs converge in both solvers. EHV 0.914–1.071 pu and 110 kV 1.013–1.025 pu (identical). Max loading 1.48, 7 overload rows (same lines: Grosskrotzenburg–Trennfeld, Daxlanden–Maximiliansau, ≈1.01 cases). Solver agreement ≤ 0.022. Bus mismatch 0.982 MW (v4final 0.919). Losses within 1 MW. 10 unsolved pandapower buses (as in v4final).

## Open gaps
- Merzen is still a radial stub: FREREN N R/X/B are not published (check the 10th JAO release).
- Parchim/Süd–Güstrow 321: 2026 status not verified (the same towers as 328 are planned for dismantling).
- 37 library-estimate circuits with partial-length JAO values. Gurtweil–Stockach rt (A853) has JAO X 347 Ω, an evident JAO error.
- HERBTG W 00016J has two Metzingen rows (node `herbentingen|380` vs the tee). Not confirmed as a duplicate. Check against class 02.
- pandapower/pandas 3 compatibility relies on the shim until pandapower supports pandas 3.

## Superseded 2021 plan (paused log of 2026-11-07, kept for reference)
Status: **paused on the coordinator's instruction before any change was applied.** The owner first
has to decide whether v5 is the 2021 grid or stays the 2026 (JAO 9th release) grid
(see issue 03 findings). No v5 model file has been written. `model_v4final.pkl` is unchanged
(sha256 b1a98e23…8ddc16, checked on cairo-lab).

## Changes
| id | asset | verdict | note |
|---|---|---|---|
| — | — | none applied | paused before the first change |

## Evidence collected (JAO Core Static Grid Model, 1st release, 2022-09, closest published state to 2021)
Downloaded to `cairo-lab:~/data/grid/v5/jao/r1/`; German lines/tielines parsed to `r1_lines.pkl`.
Planned per-row actions, if the owner picks the "2021 grid" option:

| planned id | rows | what JAO r1 shows | planned action |
|---|---|---|---|
| C1 Merzen | BRAGAU O 2414-H, merzen\|380 | Only **one** Hanekenfaehr–Westerkappeln circuit: BRAMGAU 11T0-0000-0642-D, 60.67 km, R 1.65 / X 15.26 Ω / B 264.5 µS (same values as BRAGAU W 2415-E). No Merzen circuit at all | The owner's version (a second parallel Bramgau circuit) contradicts the published 2022 model. Proposed instead: remove BRAGAU O and merzen\|380, and move its machines, load and 380/110 infeed to the nearest node in the Amprion control area. **Owner decision needed** |
| C2 | HUNTE SW 1934-R | No Ganderkesee–St. Huelfe circuit (HUNTE WS 1935-O is missing too) | remove SW. WS is a "confirm" row, so it is not touched |
| C3 | OCHSMO O/W 2066-B/2067-8 | St. Huelfe on the DUEMM S1 tee: Y-St.Huelfe legs 0996-3 (12.53 km), 0978-5 (20.38 km), 1115-Z (21.28 km) | remove OCHSMO; turn the model's direct DUEMM S1 0978-5 back into the r1 tee. Note: the model also has DS1 11T-D2-D7-00004B in parallel, which may be the same circuit counted twice (class 03) |
| C4 | BACCUM W 2250-N | single MEPPEN 11T0-0000-0640-J Hanekenfaehr–Meppen | remove W |
| C5 | GOLDGR O 2263-A, BIENWD W 2262-D | 220 kV GOLDGR 11T-D4-D7-00002U Daxlanden–Maximiliansau (2.85 km, R 0.079 / X 0.729, DLR 2320/2784 A) and BIENWD W 11T0-0000-0376-4 Maximiliansau–Mutterstadt 220 kV (47.94 km, R 2.585 / X 13.81 / B 182.5, DLR 1360/1860 A); no transformer at Maximiliansau | remove 2263-A; replace 2262-D with the two r1 circuits through a new maximiliansau\|220 pass-through bus. The rest of the 380 kV Maximiliansau area (WEINST, RGRABN, 2271-C) is also missing from r1, but those rows are not approved |
| C6 | Husum/N–Klixbuell 1860/1861, Klixbuell–Endrup 121R/122P | none in r1 (only Heide/W–Husum/N) | remove all four, remove klixbull/sud and endrup (Endrup's external injection; the DK1 exchange moves to the Kasso ties), move the injections and infeed to the nearest node in the TenneT area |
| C7 | Uckermark 481–484 | 220 kV only: 303 Bertikow–Neuenhagen, 304 Neuenhagen–Vierraden, 305 Bertikow–Pasewalk, 306 Vierraden–Pasewalk (r1 has R/X/B/L but no ratings); TR Vierraden 220/400 402/404; only PSTs 441/443 | remove 481–484; add vierraden\|220 and bertikow\|220, lines 303–306 and the two transformers. The 380 kV Pasewalk–Bertikow 407/408 and PSTs 442/444 are also missing from r1 (not approved, reported only) |
| C8 Perleberg 516 | 11TD8L516------N | r1 lists 516 **at 380 kV with the model's values** | do not apply (published values first) |
| C9 Bürs option A | BLUDNZ W 11T-D4-D7-00001W, DELLM O 11T0-0000-0954-P | (OSM arc Bürs→Werben 47.0 km) | set length 47.0 km; R/X/B = per-km values of the same circuit's far legs × 47.0 km (BLUDNZ W ≈ 0.401 Ω/km X; DELLM O ≈ 0.330 Ω/km X). The published X 9.08/9.13 Ω means ≈ 0.19 Ω/km over 47 km, which is not physical for a 220 kV overhead line. This change is valid under both options |

Not approved, so not applied: Lamspringe–Wahle and Borken–Mecklar (date not verified; r1 has Borken–Mecklar 1/2), BACCUM O, HUNTE WS, Pulgar–Vieselbach (keep), Handewitt–Jardelund (keep), BRAGAU W (keep).

## Validation setup (cairo-lab, ready but not yet run end to end)
- `~/data/grid/v5/cache/` mirrors the Mac validation inputs (ENTSO-E 2026, SMARD, energy-charts, population, compensation) with the paths changed.
- `~/data/grid/v5/cache/work/pf_v5/go4.py` is a copy of the frozen go4.py with three changes: it uses the spawn start method (fork deadlocks on Linux), it dumps the full line loadings of both solvers per case (`flows_*.pkl`), and its paths point to the cache.
- `~/data/grid/v5/run_check.sh TAG MODEL` runs it on `taskset -c 16-31`. The build step for the v4final reference succeeded (9 cases, 14 s). The solve step fails because **pandapower is not installed in cairo-lab's tranpy venv** (it is not in the Mac .venv either), so it has to be installed before the two-solver check can run.
