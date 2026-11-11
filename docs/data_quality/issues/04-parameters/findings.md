# 04 Missing or estimated parameters: findings

Status: findings ready, waiting for the owner's decision. No change was made to `model_v5.pkl` (sha256 ec0ba792…, checked) or `model_v4final.pkl` (b1a98e23…, checked).
Data: `cairo-lab:~/data/grid/v5/model_v5.pkl`, JAO Core SGM 9th release (2026-03-16, `jao/r9`) and 1st release (2022-09, `jao/r1`), https://www.jao.eu/static-grid-model.
v4.1 events: 11,159 events / 1,325 unstable (`ml_v41dyn/features`), mapped to circuits by the class-01 rule (fault_line = row among the in-operation CIM lines of v4final, without osm-4782-0; 1,015 = 1,015).
Work files: `cairo-lab:~/data/grid/v5/dq04/` (scripts, `occurrences.csv`). PF runs: `cache/work/pf_v5/dq04{herbtg,herbtg2,bursh,minor,s321}`, compared with `v5` using `summ.py`.

## Important for reading the model
The power flow (`go4.py`) uses the CIM table `ac_line_segment`, not `grid_de_circuits`. The two differ on 100 rows. The 37 "estimated" JAO circuits carry library std-type values in the CIM and older values in the store. Example: DS1 has X 16.19 Ω in the CIM and 22.86 Ω in the store. The tie lines carry whole-circuit lengths in the CIM only. All "model" values in `occurrences.csv` are CIM values.

## Occurrences (143 rows)
| group | what | rows | outcome |
|---|---|---|---|
| A | JAO circuits whose CIM R/X/B come from the std-type library instead of the JAO row (the "37": 35 rows here, plus 421 in B and DUEMM S1, which became a duplicate in v5) | 35 | 5 fix proposed, 23 keep, 7 unresolved |
| B | JAO length 0 / missing (Bürs ties, Werben stubs) | 6 | 2 keep, 4 unresolved (lengths: fix proposed inside the row) |
| C | single half-share reports doubled by the build (`_reconcile_ties`) but labelled `published` | 11 | fix proposed (label only) |
| D | OSM/curated additions with median-per-km estimates (no JAO row exists) | 61 | keep |
| E | Parchim/Süd–Güstrow 321 status | 1 | unresolved |
| F | store vs CIM length/X out of sync | 29 | fix proposed (sync only) |
Totals: **45 fix proposed, 86 keep, 12 unresolved.** Rows already flagged by other classes (zero-length rows that the model already fills: Kühmoos–Tiengen, DONAU W/O, UEB 41/42) are not repeated.

## Proposals with power-flow effect (ranked)
1. **HERBTG W tee (Hoheneck–Herbertingen–Metzingen), rows 04-018/019/024. Fix proposed, medium confidence.**
   The r9 rows are tee legs: Y-Hoheneck 0963-O (62.77 km, X 15.72 Ω), Y-Herbertingen 1879-1 (61.40 km, X 8.762, changed in r8 2025-09-01), and Y-Metzingen 00016J (Amprion 1.0 km + TransnetBW 2.71 km, "tripod – length until Y-Node").
   The model instead has: Hoheneck→Herbertingen direct, 121.7 km with library X 30.8; 1879-1 as a 61.4 km dead-end stub at `y-herbertingen#herbtg w`; and two 61.8 km dead-end stubs into Metzingen from `herbentingen|380` (a misspelt "foreign" schematic node at Hoheneck's coordinates) and from `y-hoheneck#herbtg w`. Plants are attached to these stub nodes.
   Proposal: one Y node. Hoheneck–Y gets the r9 values. 00016J becomes one leg Y–Metzingen with R 0.264, X 0.716, B 11.64 µS, 3.71 km (the sum of both shares), and its second row becomes a duplicate. The two dead nodes are removed so the plants reattach.
   PF (`dq04herbtg2`): 18/18 converge, islands unchanged, mismatch 0.74 MW. Metzingen–Wendlingen loading changes by up to +0.35. Two marginal regressions: overload rows go from 7 to 8 (Y-Neurott (-GKM - Weinheim) rt 0.998→1.002, high_load 0.1), and EHV vmin goes from 0.914 to 0.911 (high_wind 0.2). This is outside the strict v5 acceptance rule, so the owner decides.
   Without the node removal (`dq04herbtg`), 7–11 machines become islanded, so that variant is not acceptable.
   v4.1 events on the affected rows: 33 (1 unstable). Overlaps class 02 (topology first).
2. **Parchim/Süd–Güstrow 321 (04-042). Unresolved.** 321 is in neither JAO r1 nor r9; r1 lists only 328. The plan approval for the 380 kV Ersatzneubau provides for dismantling all 163 old towers alongside the new build (UVP-Verbund notice 2024-02-26; GvW press release on the PFB of 12/2023). No source states 321's 2026 status. Sensitivity run `dq04s321` (321 out of service): global metrics identical, but 332-322 Wolmirstedt–Parchim/Süd loading rises by up to +0.45 (high_solar). 18 events (0 unstable).
3. **Bürs ties 422 / 405A / 406A (+421) (04-001..006). Unresolved, owner decision.** TransnetBW's r9 rows state "total length 91.4 / 14.35 / 28.89 km, Other TSO is owner of the line", and APG reports the same R/X/B. The build summed both reports (X 28.52 / 4.48 / 8.98), so it probably counts the whole line twice. However, the single report means 0.156 Ω/km on all four circuits, which is low for a 220/380 kV overhead line, so it is not clear which value is wrong.
   Sensitivity `dq04bursh` (halved): all global metrics identical, largest loading change 0.06 on 422 itself. Proposed in any case: fill the 0 km lengths with TransnetBW's stated totals (APG gives 34.5 km for 405A, which conflicts). No PF effect; it affects only length features and the map. 18 events (3 unstable).
4. **Minor parameter replacements (`dq04minor`).** DS1 = TenneT share + Amprion share reported under DUEMM S1 (R 1.754 / X 16.026 / B 273.3 µS; the JAO change log of 2024-03-04 makes them one circuit). WESTW W1 = sum of the Amprion and TenneT shares (X 9.142 for 35.79 km). NSACHS N: the r1 full-circuit values agree with the library within 0.3 %, so it is kept.
   PF: largest loading change 0.006, global metrics identical. These are fix proposed only for provenance (published instead of library), and the gain is small. Events 40 (0 unstable).

## Kept / unresolved without PF run (reasons in the CSV)
- Keep, because the other side's share is not published or the JAO value is an evident error. Shares missing: Swissgrid and Energinet are not in Core, TransnetBW shares of RHEIN N/S and BIENWD W, TenneT share of TAUNUS 2. Evident errors or placeholders: Gurtweil–Stockach X 347 Ω, Erzhausen 1/3 X 0.05 placeholder, HUNTE SW/WS 12 km / X 0.36 Ω.
- Unresolved: FLITTA O and WIESDF (JAO per-km values look like cable sections; no source), Daxlanden–Neurott ws (JAO 13.5 km vs OSM 65.9 km), Hamburg/N–Dollern blau (50Hertz share missing), Grafenrheinfeld 411/416 (r9 2026-03-30 "temporary" Y-node; status question), RIED O tee leg (class 02).
- The 61 OSM/curated additions keep their median per-km estimates, because no published value exists. Their v4.1 events (685, 167 unstable) carry this estimate.

## Decisions for the owner
1. Approve the HERBTG W tee rebuild in spite of the marginal +1 overload row and 3 mV lower vmin? It could also be passed to class 02 first.
2. Set 321 out of service (consistent with the dismantling of the old line), or keep it until a 50Hertz statement is found?
3. Tie-line convention for the Bürs ties: keep the sum (physically plausible per km) or use the single published report (as both TSOs state)? Approve filling the lengths with TransnetBW's stated totals.
4. Approve the label-only fixes (C: 11 doubled rows → `estimated`; F: sync the store with the CIM, 29 rows) and the provenance swaps DS1 / WESTW W1.

## Not verified
- No TSO project page was found for FLITTA O, WIESDF, Daxlanden–Neurott ws, or the Bürs per-km values.
- APG's 34.5 km for 405A conflicts with TransnetBW's 14.35 km; not resolved.
- No OSM route check was done for the Bürs lengths.
- The fault_line → circuit mapping relies on the class-01 rule and on the strata/feature row order being identical.
- Rating parameters (sibling ratings, 9999 A placeholders) and the plausibility table (267 flags) are outside this pass.
