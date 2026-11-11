# v6 plan: sensitivity study on top of the frozen v5.1

v5.1 (sha cc4589c1…) stays the main case for the paper while the CCT sweep runs. v6 adds two
refinements and is first run as a **sensitivity study**. It replaces v5.1 only if the CCT
shift is large.

## Changes (one at a time, each through the gate)
1. **Converter fault behaviour: REECA1 (+ REPCA1)** on the existing REGCA1 inverters.
   - The inverters already are REGCA1 converters (wind, solar and storage, one per bus, no inertia).
   - v6 adds the WECC electrical controls: low-voltage ride-through and reactive current
     injection during faults.
   - Parameters come from the grid code (VDE-AR-N 4120/4130 fast fault current: k-factor,
     voltage dead band), never invented values.
   - Expected effect: voltage support during faults, so the CCT may be longer in high_wind and
     high_solar.
2. **Hourly line outages (ENTSO-E A78).** See
   `docs/data_quality/issues/13-hourly-outages/NOTE.md` for the data and method.
   - Expected effect: slightly shorter CCTs.

## Test
- Pre-flight and gate on v6, with the same hard checks as v5.1.
- Gate improvement first: list every line above 100 % loading by name (the old metric missed a
  210 % row).
- Sensitivity sweep: high_wind_t10, high_solar_t10 and median_load_t10. Each change is run on
  its own, then both together, and compared with the v5.1 CCT per line terminal (median and
  P90 shift, rank correlation).

## Decision
- Small shift (|median| < 20 ms and rank correlation > 0.9): report it as a robustness result
  in the paper; v5.1 stays the main case.
- Large shift: v6 becomes the main case before submission, and the owner signs off.

## Paper (now, independent of v6)
Add a limitations table to the paper:
1. no REECA1;
2. D = 0 and no PSS;
3. dynamic data from published ranges (ENTSO-E H, Kundur reactances), one machine per bus;
4. no hourly outages;
5. neighbouring grids as equivalent machines plus measured border flows;
6. HVDC as fixed injections;
7. Baltic Cable connected at Lübeck 220 kV as a stand-in.

## Offshore connection points (from C14b)

Proposal only; the frozen model v5.1 and `offshore_connections_de.csv` are not changed by C14b.
In v5.1 these register connection points are not in the curated connection table (or their
node is not a model bus), so `grid_register.attach_register` spreads their MW equally over the
landing buses of the sea (North Sea: diele, dorpen/west, emden/ost, buttel |380, 528.2 MW each
= 2,112.8 MW; Baltic: lubmin, bentwisch |380, 325.4 MW each = 650.7 MW). Proposed v6 rows:

| Farm | Register point | MW (register) | Cable | Sourced landing | Proposed model bus | Sources |
|---|---|---|---|---|---|---|
| Borkum Riffgrund 3 | DolWin5 | 958.7 | DolWin5, ±320 kV DC | Emden/Ost | emden/ost\|380 | NEP 2037/2045 (2025) 2nd draft ch. 5.3.1 p. 143; offshorewind.biz 2025-10-17 |
| He Dreiht | BWEP | 870.0 | BorWin5, ±320 kV DC | Garrel/Ost (not a model bus) | **needs owner**: Garrel/Ost sits on the new 380 kV Conneforde–Cloppenburg–Merzen line, not in v5.1 (add bus + line, or nearest cloppenburg/ost\|220, 12.5 km) | TenneT BorWin5 page; offshorewind.biz 2025-08-28 (NKT BorWin5); PFB 380 kV Conneforde–Cloppenburg–Merzen sect. 2a https://www.strassenbau.niedersachsen.de/download/193687/ |
| Windanker | OWP Windanker | 300.0 | Ostwind 3, 220 kV AC | Stilow (Brünzow), 220/380 kV (not a model bus) | **needs owner**: Stilow's 380 kV tie not sourced (nearest lubmin\|380, 8.6 km; NEP P214 Stilow–Lubmin) | 50Hertz Ostwind 3 brochure; power-technology.com (50Hertz onshore works Windanker); NEP P214 |
| EnBW Baltic 2 | Offshore Plattform Baltic 2 | 302.4 | 150 kV AC | Bentwisch (150/380 kV) | bentwisch\|380 | de.wikipedia EnBW Baltic 2; EnBW Baltic 2 technik page |
| EnBW Baltic 1 | Offshore Plattform Baltic 1 | 48.3 | AC, 77 km (150 kV per EnBW Baltic 2 shared route) | Bentwisch | bentwisch\|380 | 50Hertz project page Baltic 1; EnBW Baltic 1 technik page |
| Nordergründe | INHA | 110.7 | 155 kV AC, 32 km | Inhausen | inhausen\|220 | de.wikipedia Offshore-Windpark Nordergründe; TenneT Nordergründe page |
| Riffgat | EMDB | 113.4 | 155 kV AC, 80 km | Emden/Borßum (re-feed to Emden/Ost planned Q2 2022, completion not confirmed) | emden/ost\|380 (Borßum is fed from the Emden/Ost transformers) | de.wikipedia Offshore-Windpark Riffgat; TenneT Riffgat page; UVP-Vorprüfung Riffgat 2021 (uvp-verbund.de) |
| alpha ventus | HAMA | 60.0 | 110 kV AC, 60 km via Norderney | Hagermarsch (110 kV, not a model bus) | **needs owner**: 110 kV feed-in; upstream model bus not sourced (nearest emden/ost\|380, 29 km) | de.wikipedia alpha ventus; TenneT alpha ventus page; ArL Weser-Ems Antragskonferenz 2013 |

Effect if applied: the equal shares disappear; emden/ost +958.7 (+113.4), bentwisch +350.7,
inhausen +110.7, and 1,230 MW (He Dreiht, Windanker, alpha ventus) wait for the owner's bus
decision. Through the gate as one change; the CCT sweep is re-run afterwards.

## Missing lines (from C15)

Proposal only. The frozen model is not changed. Source: docs/data_quality/issues/15-no-model-ways (F049, F072, F104, F144).

| Line | kV | Circuits | km | Endpoints (model buses) | In service | Sources | Expected local impact |
|---|---|---|---|---|---|---|---|
| Wahle-Mecklar, section Hardegsen - Mecklar (TenneT LH-11-3040) | 380 | 2 | about 120 (230 km in total, minus Wahle-Lamspringe 70 km and Lamspringe-Hardegsen 37 km, which are in JAO r9) | `hardegsen|380` - `mecklar|380` | Inaugurated 27.11.2024; per-circuit commissioning date not confirmed | hessenschau.de 27.11.2024; topagrar (TenneT, 105 km in service); EnLAG project 6 (netzausbau.de); JAO r9 has no Hardegsen-Mecklar element (only r2297/2298 and r2392/2393) | A new N-S path in parallel to Grohnde/Würgassen-Bergshausen-Borken-Mecklar. Expect lower loading on the Würgassen and Borken-Mecklar circuits and lower angles across the Lower Saxony-Hesse cut. Check that r10 of JAO picks it up before adding it. |

Watch list (not proposed until the in-service status is confirmed): Conneforde - Cloppenburg - Merzen 380 kV (NEP TTG-P21, planned 2026), Amprion Urberach - Weinheim 380 kV, "Ostfalen-Achse" near Gleidingen.
