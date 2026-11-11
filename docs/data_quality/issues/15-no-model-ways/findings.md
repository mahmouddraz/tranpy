# Issue 15: OSM 220/380 kV ways with no model circuit (C15 triage)

Source: `~/data/grid/staging_osmgraph/web_v5_1_c14b/osm_lines.geojson` on cairo-lab, `check == no_model`, 220/380 kV, way length > 2 km: 225 ways (72 over 10 km). Contiguous ways of the same voltage that share an endpoint (within 60 m) are merged into one finding: 148 findings. The grouping and per-finding data come from `~/c15_scope.py` / `~/c15_scope.csv` on cairo-lab.

Electrical reference: JAO Core Static Grid Model, 9th release (`~/data/grid/v5/jao/r9/.../202603_Core Static Grid Model_External.xlsx`). The row numbers cited are Excel row numbers in the Lines and Tielines sheets.

## Summary

| Cause | Findings | km |
|---|---|---|
| 1 not TSO-model by design | 29 | 376.9 |
| 2 routing miss (model circuit exists, not drawn on this way) | 52 | 1040.1 |
| 3 line missing from model | 4 | 30.8 |
| 4 OSM stale/wrong | 0 | 0.0 |
| 5 needs owner | 63 | 857.8 |

No finding was set to cause 4 on its own. Stale parts (the old Ostbayernring line, the retired Frimmersdorf and Brokdorf units) are noted under causes 2 and 1.

Cause 2 here means the JAO circuit is in the frozen model, but the router left it `no_terminal`, `no_path_strict` or `not_in_network`, so it is drawn as a fallback and not on this way. Fixing it needs a terminal or join override, not a single route override.

## Map fixes

**None applied.** No `web_v5_1_c15` build was created and nothing under `halo/network` was changed. The cause 2 findings group into about 35 circuits, almost all of them `no_terminal`, mainly at tie lines and substations missing from the OSM terminal set. Each one needs its own join_override, acceptance run, layer diff and PNG pair. That was not done in this pass. Fix candidates in priority order: Etzenricht-Mechlenreuth 437/479, Röhrsdorf-Weida 573/574, Preilack/Altdöbern-Streumen 560/563, FWH-S-Urberach SHWANH, Wolmirstedt-Förderstedt 437/438, Lehrte-Erzhausen 1/3, Leupolz-Oberottmarshausen ALLG O, Würgassen Y-circuits.

## Findings

| ID | kV | km | ways | cause | note | citations |
|---|---|---|---|---|---|---|
| F001 | 220 | 147.91 | 4 | 1 | Ltg 261, 281 und 282 Ostwind 1 | Ostwind 1 Ltg 261, 281 und : Ostwind 1/2 220 kV export cables from Baltic offshore wind farms to Lubmin; offshore connection, not in CGM | 50Hertz Ostwind 1/2 project pages; no 'Ostwind' element in JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) |
| F002 | 380 | 78.28 | 4 | 2 | Leupolz - Oberottmarshausen: Leupolz - Bidingen - Oberottmarshausen (ALLG O); model circuits exist but no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r147, r148 |
| F003 | 380 | 67.18 | 1 | 5 | 305: Audorf/S - Handewitt blau/grün exist and are routed, but only 31% overlap with this 380 kV way (ref 305); unclear whether model is drawn on the old 220 kV corridor | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r2193, r2194 |
| F004 | 220 | 66.42 | 3 | 2 | Leitung 2001 | Leitung 2001 Lehrte - Hardesgen: Lehrte - Erzhausen - Hardegsen 220 kV (Ltg 2001); model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r2384, r2407 |
| F005 | 380 | 63.23 | 4 | 2 | Ltg 437 Etzenricht - Mechlenreuth | Ltg 437 Etzenricht - Mec: Ostbayernring Etzenricht-Mechlenreuth 437/479; model no_path_strict. OSM carries old+new parallel ways; old line to be dismantled after commissioning | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r2243; StMWi Bayern P46 Steckbrief; ROPf Planfeststellung Abschnitt A |
| F006 | 380 | 63.2 | 2 | 2 | Ltg 574 Röhrsdorf - Remptendorf und Ltg 575 Röhrsdorf - Weid: Röhrsdorf-Weida 573/574 and Weida-Remptendorf 575/576; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r105, r106, r133, r134 |
| F007 | 220 | 60.63 | 2 | 2 | Ludersheim - Schwandorf;Lauterhofen - Neumarkt | Ludersheim : Ludersheim-Schwandorf 223; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r2310 |
| F008 | 380 | 58.93 | 1 | 2 | Ltg 560 Preilack - Streumen und ___ Altdöbern - Streumen: Preilack-Streumen 560 / Altdöbern-Streumen 563; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r3, r92 |
| F009 | 220 | 58.56 | 8 | 2 | 220 kV | Schwanheim Ost | Schwanheim Ost + West | Schwanheim: FWH-S - Urberach SHWANH O/W; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r412, r413 |
| F010 | 220 | 52.51 | 1 | 5 | Ltg 321 und 328 Parchim Süd - Güstrow: Model has 'Ltg 321 und 328 Parchim Süd - Güstrow' (not_in_network) but JAO r9 has no such element (only Wolmirstedt - Parchim/Süd 332-322, r85); source of the model element unclear | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r85 |
| F011 | 380 | 51.54 | 1 | 2 | Ltg 445 und 446 Röhrsdorf - Hradec: Röhrsdorf-Hradec 445/446 tie; model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Tielines r4 ff (Röhrsdorf - Hradec 445/446) |
| F012 | 380 | 46.67 | 3 | 2 | Leitung 3012: Würgassen - Sandershausen/O - Bergshausen and Würgassen - Vörden - Grohnde; model no_terminal/not_in_network | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r2210, r2260, r2435, r2436 |
| F013 | 380 | 46.32 | 2 | 5 | LH-14-304: Conneforde - Cloppenburg - Merzen 380 kV (LH-14-304/325): not in JAO r9; NEP TTG-P21 gives planned commissioning 2026; in-service status of each section not confirmed | NEP Projektsteckbrief TTG-P21; NLStBV Planfeststellungsbeschlüsse 2022/2023; JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) (no element) |
| F014 | 220 | 38.35 | 4 | 5 | Anlage 313 | Anlage 313 ; HOHCK-PULVE, NWHEI-PULVE | Anlage : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F015 | 380 | 36.29 | 4 | 1 | APHSS-SIEFF;EGL-Wiesental Nord | Koppelleitung Asphard (CH) : Koppelleitung Asphard/Laufenburg (CH) - Sierentz (FR): foreign CH-FR line crossing DE territory | OSM name; JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) has no Sierentz element on DE side |
| F016 | 380 | 32.85 | 4 | 2 | Ltg 441 Etzenricht (DE) - Hradec (CZ) und 442 Etzenricht (DE: Etzenricht - Hradec 441 / Prestice 442 ties; model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Tielines r98, r99, r291, r307 |
| F017 | 380 | 32.65 | 1 | 2 | : Ostbayernring Etzenricht-Mechlenreuth 437/479; model no_path_strict. OSM carries old+new parallel ways; old line to be dismantled after commissioning | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r2243; StMWi Bayern P46 Steckbrief; ROPf Planfeststellung Abschnitt A |
| F018 | 220 | 32.4 | 1 | 5 | 2053: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F019 | 220 | 32.25 | 4 | 2 | 220 kV Kondelwald | 220 kV Kondelwald; 110 kV Cochem Nord | : Wengerohr-Weißenthurm KONDLW 220 kV (Kondelwald); model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r299 |
| F020 | 220 | 30.09 | 2 | 2 | 2319: Altenkleusheim - Garenfeld/Bixterheide - Opladen BIGGE O/W; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r162-r167 |
| F021 | 220 | 28.93 | 2 | 5 | 220 kV Wengerohr;110 kV Salmtal Süd | 220 kV Wengerohr;110kV: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F022 | 220 | 27.86 | 2 | 2 | 204: Conneforde-Fedderwarden 220 blau/rot and Fedderwarden-Maade; model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r2230, r2231, r2248, r2249 |
| F023 | 220 | 26.57 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F024 | 380 | 26.39 | 1 | 5 | 4121: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F025 | 380 | 25.54 | 5 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F026 | 220 | 25.21 | 1 | 1 | 235;236: APG 220 kV 235/236 near St. Peter (AT): foreign section | operator APG; JAO Tielines r77/78 cover only the DE-AT tie |
| F027 | 380 | 23.48 | 2 | 2 | Ltg 237 Etzenricht - Mechlenreuth sowie 112 Etzenricht - Win: Ostbayernring Etzenricht-Mechlenreuth 437/479; model no_path_strict. OSM carries old+new parallel ways; old line to be dismantled after commissioning | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r2243; StMWi Bayern P46 Steckbrief; ROPf Planfeststellung Abschnitt A |
| F028 | 380 | 23.22 | 1 | 2 | 3012: Würgassen - Sandershausen/O - Bergshausen and Würgassen - Vörden - Grohnde; model no_terminal/not_in_network | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r2210, r2260, r2435, r2436 |
| F029 | 380 | 22.71 | 4 | 5 | 4529: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F030 | 380 | 22.66 | 2 | 5 | LH-14-325: Conneforde - Cloppenburg - Merzen 380 kV (LH-14-304/325): not in JAO r9; NEP TTG-P21 gives planned commissioning 2026; in-service status of each section not confirmed | NEP Projektsteckbrief TTG-P21; NLStBV Planfeststellungsbeschlüsse 2022/2023; JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) (no element) |
| F031 | 220 | 21.46 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F032 | 380 | 20.23 | 1 | 5 | : Conneforde - Cloppenburg - Merzen 380 kV (LH-14-304/325): not in JAO r9; NEP TTG-P21 gives planned commissioning 2026; in-service status of each section not confirmed | NEP Projektsteckbrief TTG-P21; NLStBV Planfeststellungsbeschlüsse 2022/2023; JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) (no element) |
| F033 | 220 | 19.08 | 1 | 2 | : Wengerohr-Weißenthurm KONDLW 220 kV (Kondelwald); model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r299 |
| F034 | 380 | 18.68 | 3 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F035 | 220 | 18.68 | 1 | 5 | 220kV Herdecke Nord+Süd: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F036 | 380 | 18.48 | 1 | 2 | 4134: Bischofsheim - Pfungstadt RIED W / Bischofsheim-Urberach ERZHAU N; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r221, r391 |
| F037 | 380 | 18.32 | 1 | 5 | Amprion Leitung Urberach – Weinheim: Amprion Urberach - Weinheim 380 kV: not in JAO r9; construction status not confirmed | OSM name; JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) (no element) |
| F038 | 380 | 16.99 | 3 | 5 | 4222: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F039 | 380 | 16.66 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F040 | 380 | 16.34 | 2 | 2 | : Eichstetten - Vogelgrun ge (Kaiserstuhl) tie; model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Tielines r250, r215 |
| F041 | 220 | 16.1 | 2 | 1 | KB0059 | KB0092: 220 kV cables KB0059/KB0092 near Emden: offshore export cables | no JAO element; refs are cable IDs |
| F042 | 220 | 15.93 | 1 | 5 | Hoheneck West: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F043 | 380 | 15.62 | 3 | 1 | Oberbachern - Menzing | Oberbachern - Menzing;Karlsfeld - Au: Stadtwerke München 380 kV Oberbachern-Menzing (DSO) | operator SWM; no JAO element Menzing |
| F044 | 380 | 15.55 | 2 | 2 | 4201 | Kirchhellen - Mehrhoog: Doetinchem - Niederrhein DOETM SW/WS tie; model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Tielines r64, r65, r331, r332 |
| F045 | 220 | 14.84 | 1 | 5 | Ltg 386 und Ltg 385: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F046 | 220 | 14.76 | 1 | 1 | : DSO-owned 220/380 kV lines (enercity, ENERVIE, Westnetz Niehl) | operator tag; no JAO element |
| F047 | 380 | 14.55 | 1 | 2 | 380-kV-Leitung Klixbüll - Bundesgrenze DK, LH-13-322: Klixbüll/S - Endrup 1/2 (Westküstenleitung to DK); model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Tielines r334, r335 |
| F048 | 380 | 13.43 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F049 | 380 | 13.25 | 3 | 3 | LH-11-3040: Wahle-Mecklar 380 kV (LH-11-3040), section Hardegsen - Mecklar: in service (inaugurated 27.11.2024), absent from JAO r9 (r9 holds only Wahle-Lamspringe 3/4 r2297/8 and Lamspringe-Hardegsen 4/5 r2392/3) | hessenschau.de 27.11.2024 'Neue Stromtrasse Wahle-Mecklar offiziell eingeweiht'; topagrar 'TenneT nimmt 105 km ... in Betrieb'; JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) (no Hardegsen-Mecklar element) |
| F050 | 220 | 13.23 | 1 | 1 | 220 kV Anilin 1A|2B: BASF Anilin 220 kV industrial line | no JAO element 'Anilin'; JAO BASF lines r362/r395 are separate |
| F051 | 380 | 13.1 | 1 | 2 | Ltg 437 und 438 Wolmirstedt - Förderstedt: Wolmirstedt-Förderstedt 437/438; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r136, r137 |
| F052 | 380 | 12.86 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F053 | 380 | 12.74 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F054 | 380 | 12.72 | 4 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F055 | 220 | 12.31 | 2 | 2 | Gemeinschaftsleitung Transnet BW / MVV Netze | Transnet BW L: GKM - Neurott - Weinheim rt (7600/4505/4504); model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r2514, r2523, r2533 |
| F056 | 380 | 12.11 | 1 | 5 | Meppen-Hanekenfähr: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F057 | 220 | 12.07 | 1 | 5 | 211;133: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F058 | 380 | 11.84 | 1 | 2 | Ltg 437 und 438 Wolmirstedt - Förderstedt: Wolmirstedt-Förderstedt 437/438; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r136, r137 |
| F059 | 380 | 11.51 | 1 | 2 | Ltg 574 Röhrsdorf - Remptendorf und Ltg 573 Weida - Remptend: Röhrsdorf-Weida 573/574 and Weida-Remptendorf 575/576; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r105, r106, r133, r134 |
| F060 | 380 | 11.49 | 2 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F061 | 220 | 11.28 | 3 | 5 | Anlage 5230 | Großgartach - Obrigheim 1x 220 kV: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F062 | 220 | 11.25 | 3 | 5 | Elsenz: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F063 | 220 | 11.07 | 1 | 5 | Hoheneck West: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F064 | 220 | 10.71 | 1 | 2 | Pfalzwerke Leitung Rhein Nord/Süd: Mutterstadt - GKM RHEIN N/S tie; model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Tielines r43, r44 |
| F065 | 380 | 10.68 | 1 | 2 | Ltg 437 und 438 Wolmirstedt - Förderstedt: Wolmirstedt-Förderstedt 437/438; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r136, r137 |
| F066 | 380 | 10.37 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F067 | 380 | 10.1 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F068 | 380 | 9.9 | 1 | 2 | 306: Diele - Meeden 380 schwarz/weiss tie; model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Tielines r277, r278, r329, r330 |
| F069 | 220 | 9.86 | 1 | 1 | : DSO-owned 220/380 kV lines (enercity, ENERVIE, Westnetz Niehl) | operator tag; no JAO element |
| F070 | 380 | 9.68 | 1 | 5 | Ostfalen-Achse: 'Ostfalen-Achse' 380 kV near Gleidingen: not in JAO r9; status not confirmed | OSM name; JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) |
| F071 | 380 | 9.66 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F072 | 380 | 9.59 | 2 | 3 | Wahle–Mecklar: Wahle-Mecklar 380 kV (LH-11-3040), section Hardegsen - Mecklar: in service (inaugurated 27.11.2024), absent from JAO r9 (r9 holds only Wahle-Lamspringe 3/4 r2297/8 and Lamspringe-Hardegsen 4/5 r2392/3) | hessenschau.de 27.11.2024 'Neue Stromtrasse Wahle-Mecklar offiziell eingeweiht'; topagrar 'TenneT nimmt 105 km ... in Betrieb'; JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) (no Hardegsen-Mecklar element) |
| F073 | 380 | 9.24 | 1 | 2 | : Handewitt - Kassø 1/2 tie; model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Tielines r288, r289 |
| F074 | 380 | 9.14 | 3 | 2 | : Dauersberg - Eiserfeld SIEGRL O/W 220 kV; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r415, r416 |
| F075 | 220 | 9.07 | 1 | 5 | Gemeinschaftsleitung Transnet BW / Netze BW: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F076 | 380 | 8.97 | 2 | 5 | : Conneforde - Cloppenburg - Merzen 380 kV (LH-14-304/325): not in JAO r9; NEP TTG-P21 gives planned commissioning 2026; in-service status of each section not confirmed | NEP Projektsteckbrief TTG-P21; NLStBV Planfeststellungsbeschlüsse 2022/2023; JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) (no element) |
| F077 | 380 | 8.92 | 1 | 2 | : Simbach - St. Peter / Altheim 233/230, 234/230; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r2422, r2423; Tielines r77, r78, r319, r320 |
| F078 | 220 | 8.9 | 1 | 5 | 220kV Herdecke Nord+Süd: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F079 | 380 | 8.36 | 1 | 2 | 4512: Koblenz - Y-Koblenz SOONWD O; model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r419-r421 |
| F080 | 380 | 8.33 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F081 | 380 | 7.77 | 2 | 2 | Vigy - Ensdorf (380kV) | Vigy - Ensdorf (380kV);Altforweiler: Ensdorf - Vigy VIGY1/2 and Ensdorf - St. Avold ties; model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Tielines r57-r59, r208, r209, r213 |
| F082 | 380 | 7.77 | 2 | 1 | Ltg 461 und 462 Goldisthal - Altenfeld | Ltg 461 und 462 Gol: Goldisthal pumped-storage plant lines 461/462 | no JAO element Goldisthal |
| F083 | 220 | 7.76 | 1 | 1 | 220 kV Frimmersdorf 4d: RWE/Amprion power-plant unit lines (Scholven, Bergkamen, Neurath, Frimmersdorf, Niederaußem) | unit names in OSM; no JAO element (Frimmersdorf units retired, see BNetzA Kraftwerksliste) |
| F084 | 220 | 7.61 | 1 | 2 | 2340: Ensdorf - Vigy VIGY1/2 and Ensdorf - St. Avold ties; model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Tielines r57-r59, r208, r209, r213 |
| F085 | 380 | 7.24 | 2 | 5 | : Conneforde - Cloppenburg - Merzen 380 kV (LH-14-304/325): not in JAO r9; NEP TTG-P21 gives planned commissioning 2026; in-service status of each section not confirmed | NEP Projektsteckbrief TTG-P21; NLStBV Planfeststellungsbeschlüsse 2022/2023; JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) (no element) |
| F086 | 380 | 7.22 | 1 | 5 | 546: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F087 | 380 | 7.07 | 1 | 2 | Suedostlink;Ltg 437 und 438 Wolmirstedt - Förderstedt: Wolmirstedt-Förderstedt 437/438; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r136, r137 |
| F088 | 380 | 7.0 | 1 | 1 | 380kV Bergkamen A: RWE/Amprion power-plant unit lines (Scholven, Bergkamen, Neurath, Frimmersdorf, Niederaußem) | unit names in OSM; no JAO element (Frimmersdorf units retired, see BNetzA Kraftwerksliste) |
| F089 | 380 | 6.94 | 1 | 2 | Ltg 560 Preilack - Streumen und ___ Altdöbern - Streumen: Preilack-Streumen 560 / Altdöbern-Streumen 563; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r3, r92 |
| F090 | 380 | 6.88 | 1 | 5 | Amprion Leitung Urberach – Weinheim: Amprion Urberach - Weinheim 380 kV: not in JAO r9; construction status not confirmed | OSM name; JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) (no element) |
| F091 | 220 | 6.67 | 1 | 5 | 220 kV Sirzenich – Bundesgrenze: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F092 | 220 | 6.64 | 2 | 5 | 2339: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F093 | 380 | 6.44 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F094 | 380 | 6.41 | 2 | 1 | Ltg MN und MP Boxberg - Bärwalde | Ltg MQ und MR Boxberg - B: LEAG power-station lines Boxberg-Bärwalde / Jänschwalde-Preilack (Ltg MA..MR) | plant line letters, no JAO element; JAO r11/12 Bärwalde-Streumen, r44/45 Graustein-Preilack are the grid lines |
| F095 | 380 | 6.35 | 1 | 1 | Ltg ME und MF Jänschwalde - Preilack: LEAG power-station lines Boxberg-Bärwalde / Jänschwalde-Preilack (Ltg MA..MR) | plant line letters, no JAO element; JAO r11/12 Bärwalde-Streumen, r44/45 Graustein-Preilack are the grid lines |
| F096 | 220 | 6.31 | 1 | 2 | 5130: Gurtweil - Laufenburg ge / Y-Laufenburg rtge ties; model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Tielines r253, r270 |
| F097 | 380 | 6.29 | 1 | 1 | Ltg MC und MD Jänschwalde - Preilack: LEAG power-station lines Boxberg-Bärwalde / Jänschwalde-Preilack (Ltg MA..MR) | plant line letters, no JAO element; JAO r11/12 Bärwalde-Streumen, r44/45 Graustein-Preilack are the grid lines |
| F098 | 380 | 6.26 | 1 | 1 | : Schluchseewerk AG pumped-storage plant lines at Kühmoos | operator Schluchseewerk; no JAO element |
| F099 | 380 | 6.22 | 1 | 1 | Ltg MA und MB Jänschwalde - Preilack: LEAG power-station lines Boxberg-Bärwalde / Jänschwalde-Preilack (Ltg MA..MR) | plant line letters, no JAO element; JAO r11/12 Bärwalde-Streumen, r44/45 Graustein-Preilack are the grid lines |
| F100 | 380 | 6.16 | 1 | 2 | : Conneforde-Fedderwarden 220 blau/rot and Fedderwarden-Maade; model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r2230, r2231, r2248, r2249 |
| F101 | 220 | 5.83 | 1 | 2 | Amprion Leitung Rheinau Ost / West: GKM - Neurott - Weinheim rt (7600/4505/4504); model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r2514, r2523, r2533 |
| F102 | 380 | 5.74 | 1 | 5 | 380kV Stummhafen 1A + ANK: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F103 | 380 | 5.64 | 2 | 5 | : Conneforde - Cloppenburg - Merzen 380 kV (LH-14-304/325): not in JAO r9; NEP TTG-P21 gives planned commissioning 2026; in-service status of each section not confirmed | NEP Projektsteckbrief TTG-P21; NLStBV Planfeststellungsbeschlüsse 2022/2023; JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) (no element) |
| F104 | 380 | 5.52 | 1 | 3 | Wahle–Mecklar: Wahle-Mecklar 380 kV (LH-11-3040), section Hardegsen - Mecklar: in service (inaugurated 27.11.2024), absent from JAO r9 (r9 holds only Wahle-Lamspringe 3/4 r2297/8 and Lamspringe-Hardegsen 4/5 r2392/3) | hessenschau.de 27.11.2024 'Neue Stromtrasse Wahle-Mecklar offiziell eingeweiht'; topagrar 'TenneT nimmt 105 km ... in Betrieb'; JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) (no Hardegsen-Mecklar element) |
| F105 | 380 | 5.44 | 1 | 5 | Ltg 3845 (System 301): No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F106 | 380 | 5.29 | 1 | 5 | 4201: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F107 | 380 | 4.91 | 1 | 1 | Ltg 431 und Ltg 432 Schkopau - Lauchstädt: Schkopau power-plant lines 431/432 to Lauchstädt | no JAO element Schkopau |
| F108 | 380 | 4.9 | 1 | 1 | Koppelleitung Laufenburg (CH) - Sierentz (F): Koppelleitung Asphard/Laufenburg (CH) - Sierentz (FR): foreign CH-FR line crossing DE territory | OSM name; JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) has no Sierentz element on DE side |
| F109 | 380 | 4.61 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F110 | 380 | 4.36 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F111 | 380 | 4.29 | 1 | 2 | : Dauersberg - Eiserfeld SIEGRL O/W 220 kV; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r415, r416 |
| F112 | 220 | 4.24 | 1 | 2 | 2319: Altenkleusheim - Garenfeld/Bixterheide - Opladen BIGGE O/W; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r162-r167 |
| F113 | 380 | 4.14 | 1 | 1 | Koppelleitung Laufenburg (CH) - Sierentz (F): Koppelleitung Asphard/Laufenburg (CH) - Sierentz (FR): foreign CH-FR line crossing DE territory | OSM name; JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) has no Sierentz element on DE side |
| F114 | 380 | 3.96 | 1 | 1 | 380 kV Neurath 4d: RWE/Amprion power-plant unit lines (Scholven, Bergkamen, Neurath, Frimmersdorf, Niederaußem) | unit names in OSM; no JAO element (Frimmersdorf units retired, see BNetzA Kraftwerksliste) |
| F115 | 380 | 3.95 | 1 | 5 | 3100 Dollern - Sottrum: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F116 | 220 | 3.8 | 1 | 2 | : Altenkleusheim - Garenfeld/Bixterheide - Opladen BIGGE O/W; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r162-r167 |
| F117 | 380 | 3.79 | 1 | 2 | Ltg 415 Vieselbach - Großschwabhausen - Remptendorf: Vieselbach - Großschwabhausen - Remptendorf 415 | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r129 |
| F118 | 380 | 3.74 | 1 | 2 | 4338: Gronau - Hengelo HENGL SW/WS tie; model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Tielines r66, r67 |
| F119 | 220 | 3.65 | 1 | 2 | : Altenkleusheim - Garenfeld/Bixterheide - Opladen BIGGE O/W; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r162-r167 |
| F120 | 220 | 3.64 | 1 | 5 | Ltg 293 und 294: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F121 | 380 | 3.6 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F122 | 380 | 3.54 | 1 | 1 | Scholven B+D+E: RWE/Amprion power-plant unit lines (Scholven, Bergkamen, Neurath, Frimmersdorf, Niederaußem) | unit names in OSM; no JAO element (Frimmersdorf units retired, see BNetzA Kraftwerksliste) |
| F123 | 380 | 3.38 | 1 | 1 | 380kV Scholven F: RWE/Amprion power-plant unit lines (Scholven, Bergkamen, Neurath, Frimmersdorf, Niederaußem) | unit names in OSM; no JAO element (Frimmersdorf units retired, see BNetzA Kraftwerksliste) |
| F124 | 380 | 3.37 | 1 | 5 | 4230: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F125 | 220 | 3.23 | 1 | 5 | 220 kV Heisdorf - Quint Nord;Niederstedem West: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F126 | 380 | 3.22 | 1 | 2 | : Garenfeld - Kruckel OSTENDE/WESTENDE; model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r360, r477 |
| F127 | 380 | 3.06 | 1 | 1 | 308 AKW Brokdorf - UW Wilster: Brokdorf NPP connection (plant shut 31.12.2021) | BNetzA Kraftwerksliste; JAO has only Brokdorf 220 kV grid lines |
| F128 | 220 | 3.05 | 1 | 2 | 220 kV Rheinau Ost|West: GKM - Neurott - Weinheim rt (7600/4505/4504); model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r2514, r2523, r2533 |
| F129 | 220 | 3.02 | 1 | 1 | 220 kV Niederaußem 2b; Rommerskirchen Ost: RWE/Amprion power-plant unit lines (Scholven, Bergkamen, Neurath, Frimmersdorf, Niederaußem) | unit names in OSM; no JAO element (Frimmersdorf units retired, see BNetzA Kraftwerksliste) |
| F130 | 380 | 3.01 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F131 | 220 | 2.96 | 1 | 1 | Scholven C: RWE/Amprion power-plant unit lines (Scholven, Bergkamen, Neurath, Frimmersdorf, Niederaußem) | unit names in OSM; no JAO element (Frimmersdorf units retired, see BNetzA Kraftwerksliste) |
| F132 | 220 | 2.9 | 1 | 2 | 220 kV Kondelwald; 110 kV Pünderich: Wengerohr-Weißenthurm KONDLW 220 kV (Kondelwald); model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r299 |
| F133 | 380 | 2.84 | 1 | 1 | 380 kV Niehl 3: DSO-owned 220/380 kV lines (enercity, ENERVIE, Westnetz Niehl) | operator tag; no JAO element |
| F134 | 380 | 2.8 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F135 | 380 | 2.73 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F136 | 220 | 2.7 | 1 | 2 | : Dauersberg - Eiserfeld SIEGRL O/W 220 kV; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r415, r416 |
| F137 | 220 | 2.69 | 1 | 2 | : Altenkleusheim - Garenfeld/Bixterheide - Opladen BIGGE O/W; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r162-r167 |
| F138 | 380 | 2.68 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F139 | 380 | 2.56 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F140 | 220 | 2.53 | 1 | 5 | 220 kV Niederstedem Ost|West: No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F141 | 220 | 2.49 | 1 | 2 | : Simbach - St. Peter / Altheim 233/230, 234/230; model no_path_strict | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Lines r2422, r2423; Tielines r77, r78, r319, r320 |
| F142 | 220 | 2.48 | 1 | 1 | : Schluchseewerk AG pumped-storage plant lines at Kühmoos | operator Schluchseewerk; no JAO element |
| F143 | 380 | 2.46 | 1 | 1 | Frimmersdorf 5e,Frimmersdorf 6f: RWE/Amprion power-plant unit lines (Scholven, Bergkamen, Neurath, Frimmersdorf, Niederaußem) | unit names in OSM; no JAO element (Frimmersdorf units retired, see BNetzA Kraftwerksliste) |
| F144 | 380 | 2.4 | 1 | 3 | Wahle–Mecklar: Wahle-Mecklar 380 kV (LH-11-3040), section Hardegsen - Mecklar: in service (inaugurated 27.11.2024), absent from JAO r9 (r9 holds only Wahle-Lamspringe 3/4 r2297/8 and Lamspringe-Hardegsen 4/5 r2392/3) | hessenschau.de 27.11.2024 'Neue Stromtrasse Wahle-Mecklar offiziell eingeweiht'; topagrar 'TenneT nimmt 105 km ... in Betrieb'; JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) (no Hardegsen-Mecklar element) |
| F145 | 380 | 2.25 | 1 | 2 | : Hagenwerder - Mikułowa 567/568 tie; model no_terminal | JAO Core SGM r9 (202603_Core Static Grid Model_External.xlsx) Tielines r17, r18, r198, r199 |
| F146 | 220 | 2.24 | 1 | 1 | Frimmersdorf 1a, Frimmersdorf 2b, Frimmersdorf 3c, Frimmersd: RWE/Amprion power-plant unit lines (Scholven, Bergkamen, Neurath, Frimmersdorf, Niederaußem) | unit names in OSM; no JAO element (Frimmersdorf units retired, see BNetzA Kraftwerksliste) |
| F147 | 380 | 2.04 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |
| F148 | 220 | 2.03 | 1 | 5 | : No independent source matched; inconclusive | JAO r9 searched by station names near endpoints: no unambiguous element |

## Needs owner

F003 (67.18 km), F010 (52.51 km), F013 (46.32 km), F014 (38.35 km), F018 (32.4 km), F021 (28.93 km), F023 (26.57 km), F024 (26.39 km), F025 (25.54 km), F029 (22.71 km), F030 (22.66 km), F031 (21.46 km), F032 (20.23 km), F034 (18.68 km), F035 (18.68 km), F037 (18.32 km), F038 (16.99 km), F039 (16.66 km), F042 (15.93 km), F045 (14.84 km), F048 (13.43 km), F052 (12.86 km), F053 (12.74 km), F054 (12.72 km), F056 (12.11 km), F057 (12.07 km), F060 (11.49 km), F061 (11.28 km), F062 (11.25 km), F063 (11.07 km), F066 (10.37 km), F067 (10.1 km), F070 (9.68 km), F071 (9.66 km), F075 (9.07 km), F076 (8.97 km), F078 (8.9 km), F080 (8.33 km), F085 (7.24 km), F086 (7.22 km), F090 (6.88 km), F091 (6.67 km), F092 (6.64 km), F093 (6.44 km), F102 (5.74 km), F103 (5.64 km), F105 (5.44 km), F106 (5.29 km), F109 (4.61 km), F110 (4.36 km), F115 (3.95 km), F120 (3.64 km), F121 (3.6 km), F124 (3.37 km), F125 (3.23 km), F130 (3.01 km), F134 (2.8 km), F135 (2.73 km), F138 (2.68 km), F139 (2.56 km), F140 (2.53 km), F147 (2.04 km), F148 (2.03 km)

Web sources used: hessenschau.de (27.11.2024, Wahle-Mecklar inauguration); topagrar.com (TenneT puts 105 km into service); netzentwicklungsplan.de TTG-P21 and P46 project sheets; strassenbau.niedersachsen.de CCM planning approvals 2022/2023; stmwi.bayern.de Ostbayernring P46 sheet; ropf.bayern.de Ostbayernring approval.
