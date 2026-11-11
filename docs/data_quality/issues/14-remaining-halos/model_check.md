# 14 model check: class (c) proposals A016-A072 (candidates and gates, not applied)

Date: 2026-11-09. Reference: frozen `v5/frozen/model_v5_frozen.pkl` (sha 3393038e), gate reference
`s3393038e` (default flags). Frozen folder untouched. No CHANGELOG entry and no re-freeze: **the owner decides.**
Candidates: cairo-lab `~/data/grid/v5/step5/` (`fix5.py IN OUT STEP`, one fix per copy, `*.changes.csv`
logs every field), gates in `step5/gates/`, dynamic cases `step5/dyn_{frozen,all}`, CCT `step5/cct_{frozen,all}`.

Owner rule (2026-11-09, applied here): a change needs evidence independent of OSM and of our model
(JAO r9, TSO/plant pages, BNetzA/NEP, planning documents, imagery). OSM-only or inconclusive evidence -> **keep**.

## Verdicts

| # | element | finding | independent evidence | strength | verdict |
|---|---|---|---|---|---|
| 1 A016 | osm-14135-0/1 Sechtem - Brauweiler 380 kV (export, est. X 5.64 Ω each) | Built from one OSM way 24431541 (ref 4511, circuits=2, cables=6), the tower line of r9 VILLE O/W; route overlaps VILLE O/W, VORGBG O at 95 % | JAO Core SGM r9 (2026-03-16 publication, Lines sheet): Sechtem has 9 Amprion 380 kV rows (Meckenheim, Liessem, Weissenthurm, Siegburg, SECHTM N/S, Dauersberg, VILLE O/W, VORGBG O, Knapsack); Brauweiler 380 kV has exactly `Brauweiler - Rommerskirchen BRAUWL W 11T0-0000-0927-S` and `Brauweiler - Knapsack BURBAC O 11T0-0000-0928-P`. No Sechtem - Brauweiler element in r9 or r1, and no Change Log row adding/removing one | conclusive (JAO, two releases) | **fix: status `duplicate`** (candidate c1) |
| 2 A031 | osm-5010-0 Garenfeld - Y-Garenfeld 220 kV (ENERVIE) | OSM way 284701565 "220kV/d Nordleitung+Südleitung" (2 systems); stored r9 routes BIGGE W leg / BIXTRH N overlap the way only 18-20 % | JAO r9 lists only TSO elements; the ENERVIE 220 kV lines are not in JAO. Duplicate not shown | none | **keep** |
| 3 A033 | osm-4706-1-0/1 junction - Stahlwerk Thüringen | OSM way 812686808 "Ltg 386" cables=3 (one system), model 2 | none independent of OSM (50Hertz 220 kV Ltg 383-386 not in JAO r9/r1; no TSO page found) | partial (OSM only) | **keep** (candidate c3 gated for information) |
| 4 A042 | curated Zwönitz - Röhrsdorf 577/578 + osm-14312 | chains are in series, not parallel (no double path). Curated length 30.6 km vs OSM route 24.54 km; spur Niederzwönitz - Zwönitz carries 2 systems in OSM while the model loops both in | JAO r9 has no 577/578/Markersbach/Zwönitz row; web search (50Hertz, NEP, LD Sachsen) found no topology or length | none | **keep** (+ KNOWN_LIMITS); candidate c4 (length only) gated for information |
| 5 A059 | osm-4291-0..3 Niederaußem 220 kV | shared way 24841695 "1a\|2b\|3c\|4d", cables=12 = 4 systems; model 4 to one plant bus (2+2 split is electrically identical with one plant bus) | none | none | **keep** |
| 6 A072 | osm-5127-0..2 Schwelgern | way 425103271 circuits=3, cables=12; model 3 equals the tag. Halo piece lies on neighbour way 425103273 (circuits=2): map artefact | none | none | **keep** |
| 7 A060 | osm-4386-0 Schwelgern/Beeck | OSM descriptions "220kV/s Beeck Tr211+212" on 4 ways suggest 2 systems; model 1 | none independent (industrial/Amprion-Westnetz feeder, not in JAO) | partial (OSM only) | **keep** (c7 gated for information) |
| 8 A061 | osm-4306-0 LMG Ost+West | OSM name/circuits=2; model 1 | none independent (RWE plant line) | partial (OSM only) | **keep** (c8 gated) |
| 9 A063 | osm-4329-0 Nordleitung+Südleitung | OSM circuits=2, "220kV/d"; model 1 | none independent (ENERVIE line) | partial (OSM only) | **keep** (c9 gated) |

## Gates (each candidate alone vs s3393038e, then combined)

All PASS on the hard checks: 18/18 convergence in both solvers, islands 9 = 9, pp NaN buses 8 = 8,
machines cut off 0, store/CIM in sync. Overloaded rows 10 = 10, vmin 0.918 = 0.918 (both solvers), maxl 1.49 =.

| cand | sha | losses Δ sy/pp (MW, 18 runs) | lines > 0.10 loading change (all within 1 hop of the change) |
|---|---|---|---|
| c1 | cc4589c1 | +18 / +19 | BRAUWL W -0.11..-0.21 (16 rows), WABERG W +0.12..+0.13 (8), BURBAC O +0.11..+0.12 (4); osm-14135 removed |
| c3 | bdb754fb | 0 / +1 | none (osm-4706-1-1 removed) |
| c4 | f0c024c4 | -1 / -1 | none |
| c7 | 21d7792f | -9 / -3 | osm-4386-0 -0.11..-0.43 (flow shared with the new twin) |
| c8 | 4ca6e4bf | 0 / 0 | none (new twin only) |
| c9 | 1202b8ec | -3 / -3 | osm-4329-0 -0.10..-0.24 (flow shared with the new twin) |
| all | 50a300b2 | +9 / +13 | union of the above, nothing elsewhere |

No side effect beyond the intended area in any candidate. c1's changes are the three 380 kV lines that
take over the Sechtem - Brauweiler flow (expected).

## TSA effect

Lines within 2 hops (lines touching a terminal or its direct neighbour bus), v5 model / v4.1 fault events
(11,162 events, 18 cases): c1 47 / 517, c3 6 / 62, c4 32 / 319, c7 13 / 166, c8 13 / 150, c9 17 / 185;
any change: 1,374 events (12 %).

CCT bracket (combined candidate vs frozen; F1b `entsoe_kundur`, shift_governor_limits=True, outages off;
v5pre.py + cct_bracket_v5.py, bracket width 7.5 ms; s), position 0 = from bus, 1 = to bus:

| case | line | pos | frozen | combined |
|---|---|---|---|---|
| high_load_t10 | BRAUWL W Brauweiler - Rommerskirchen | 0 | 0.406-0.414 | **0.473-0.481** |
| high_load_t10 | BRAUWL W | 1 | 0.339-0.347 | 0.339-0.347 |
| high_load_t10 | VILLE O Rommerskirchen - Sechtem | 0 | 0.339-0.347 | 0.339-0.347 |
| high_load_t10 | VILLE O | 1 | 0.369-0.377 | **0.399-0.406** |
| high_load_t10 | WABERG W Knapsack - Sechtem | 0 | 0.317-0.325 | 0.317-0.325 |
| high_load_t10 | WABERG W | 1 | 0.377-0.384 | **0.406-0.414** |
| high_load_t10 | Zwönitz - Röhrsdorf 577 | 0 | failed (numerical) | failed (numerical) |
| high_load_t10 | Zwönitz - Röhrsdorf 577 | 1 | 0.325-0.332 | 0.325-0.332 |
| low_load_t10 | BRAUWL W | 0 | 0.399-0.406 | **0.451-0.458** |
| low_load_t10 | BRAUWL W | 1 | 0.295-0.302 | 0.295-0.302 |
| low_load_t10 | VILLE O | 0 | 0.295-0.302 | 0.295-0.302 |
| low_load_t10 | VILLE O | 1 | 0.362-0.369 | **0.384-0.391** |
| low_load_t10 | WABERG W | 0 | 0.339-0.347 | 0.339-0.347 |
| low_load_t10 | WABERG W | 1 | 0.369-0.377 | **0.384-0.391** |
| low_load_t10 | Zwönitz - Röhrsdorf 577 | 0 | above 1.0 | above 1.0 |
| low_load_t10 | Zwönitz - Röhrsdorf 577 | 1 | failed (numerical) | failed (numerical) |

Faults at the Sechtem/Brauweiler-side terminals shift by +15 to +67 ms (2-9 bracket widths); Zwönitz unchanged.
The shift comes from c1 (the only change near these lines; removing the phantom Sechtem - Brauweiler pair
changes the post-fault path). c1 was not CCT-tested alone; the other changes are > 2 hops from these lines.

## Recommendation

The effect is not negligible: c1 moves CCT brackets beyond the bracket width around Sechtem/Brauweiler,
and c1 is the only proposal with conclusive independent evidence (JAO r9). Recommendation:
**re-freeze as v5.1 with c1 only** (candidate `step5/cand_c1.pkl`, sha cc4589c1, gate PASS), and keep
#2-#9 with KNOWN_LIMITS entries (OSM-only circuit counts at non-TSO 220 kV feeders: Stahlwerk Thüringen,
Beeck, LMG, Nordleitung+Südleitung; Zwönitz topology/length unpublished). If the owner prefers not to
re-freeze, the alternative is a KNOWN_LIMITS entry for osm-14135 quoting the CCT shift above.
