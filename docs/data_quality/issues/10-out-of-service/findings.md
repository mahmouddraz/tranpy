# 10 Circuits flagged "Out-of-service" in the JAO change log but listed and in operation (detection only)

This is a read-only check; nothing was changed. The model is `model_v5_4a.pkl` (sha 75e6058b…).

Sources:
- JAO Core SGM r9 (2026-03): Lines, Tielines and the Change Log, which covers r3 (2023-03) to r9.
- JAO r1 (2022-09), via `jao/r1_lines.pkl`.
- v4.1 event counts from `dq04/events_by_mrid.csv`.
- OSM route km from agent A's `network/edge_check.csv`.

Every row is in `occurrences.csv`. Each row carries the full change-log history, r1/r9 presence, OSM km, events, the verdict, the evidence and a confidence.

## Selection
There are 27 model circuits with status `in_operation` whose r9 Change Log contains an "Out-of-service" entry. The pattern behind them:
- 25 of the 27 are TenneT circuits, almost all flagged in one batch in v6 (2024-08-30). The v6 batch has 28 TenneT "Out-of-service." rows in total.
- r9 still lists every one of them. They are not marked in the sheets, and the Comment column is empty.
- TenneT does log reversals elsewhere. Examples are v5 "TR Conneforde/O T423 Back in operation" and the v3 "Removed" entries followed by v4 "New line". No reversal appears for the circuits below.
- In v8 (2025-09-01), TenneT removed four of the v6-flagged circuits with "Existing line was removed from the model": Isar–Ottenhofen 444, Isar–Pleinting 452, Grohnde–Würgassen 2 and Grosskrotzenburg–Karben 1. Two of those are still in the model. So "removed" marks a permanent end. The circuits that stay listed with only a v6 flag look like outages that were temporary in the v6 reference period (mid-2024). Their state in Nov 2025–Oct 2026 is not published.

## Verdicts
| verdict | n |
|---|---|
| keep: the flag was reversed (v3 "Removed", then v4 "New line") | 3 (Elsfleth/W–Ganderkesee rot, Audorf–Audorf/S T421, Ganderkesee–Niedervieland schwarz) |
| propose out_of_service | 6 (below) |
| temporarily out at v6, state in the v5 window not published (keep `in_operation`, flag `review`) | 18 |

### Proposed status changes (go to the owner; approved ones go through gate.sh via the data agent)
| # | circuit | EIC | evidence | v4.1 events | confidence |
|---|---|---|---|---|---|
| 1 | Grohnde–Würgassen 2 | 11TD2L000000092X | v6 OOS; v8 2025-09-01 "Existing line was removed from the model"; no OSM route (0 km) | 2 | medium |
| 2 | Grosskrotzenburg–Karben 1 | 11TD2L0000003128 | v6 OOS; v8 2025-09-01 "Existing line was removed from the model" (OSM still maps 19 km) | 9 | medium |
| 3 | Hradec–Etzenricht 441 (CEPS tie) | 10T-CZ-DE-00001W | v6 OOS; v8 2025-09-01 "Decommissioned"; no OSM route | 14 | medium |
| 4–6 | EOWS tripod Y-Meppen / Y-Doerpen/W / Y-Niederlangen (-Doerpen/W - Meppen) | 11T0-0000-0983-G, -0990-L, -0991-I | v6 OOS; v7 2025-03-01 adds "Doerpen West - Meppen EMSLD WB" (11T0-0000-2116-Q) on the same corridor, so the tripod looks superseded. Check that the two are not the same circuit counted twice | 13 / 9 / 12 | medium |

The r9 sheet contradicts the change log for rows 1–3: the circuits are still listed after "removed"/"Decommissioned". I treat the change log as the dated statement. In all six cases the change would apply to the whole v5 window (Nov 2025–Oct 2026) and to all v5 hours.

### Temporarily out at v6, no published state for Nov 2025–Oct 2026 (no change proposed)
These keep `in_operation`. I propose a `review` flag and an owner note.

| circuit | evidence | v4.1 events |
|---|---|---|
| Isar–Pleinting 451 | TenneT notice 2023-05-11 (LH-08-B117): "Erneuerung der Leiterseile und Isolatorenketten", same conductor cross-section; no completion date. 452 is permanently gone (v8) | 13 |
| Isar–Ottenhofen 443 | TenneT info sheet 2024-03-07 "Generalsanierung Ottenhofen–Isar" (refurbishment, stays 380 kV); no outage dates. 444 removed v8 | 10 |
| Irsching–Ottenhofen 421 | NEP P54 corridor | 11 |
| Grafenrheinfeld–Raitersaich 433 | TenneT maintenance on the line since 2022-09 (Bayernwerk/TenneT press release 2022-11); 434 in service | 14 |
| Borken–Giessen/N 2 | NEP TTG-P133 re-conductoring, 73 km to 4,000 A, planned commissioning 2027. The likeliest of these to be out for part of the window, but no months are published | 9 |
| Bergrheinfeld/W–Grafenrheinfeld 426 | 3 km station link; 427 and 487 are parallel | 18 |
| Kummerfeld–Dollern grün | – | 8 |
| Hamburg/N–Lübeck gelb | – | 12 |
| Stadorf–Lüneburg 1 | – | 13 |
| Voslapp–Inhausen rot | – | 7 |
| Klein Ilsede–Mehrum/N 1 | Mehrum/N area; the 220 kV stub is already dead (06 #5) | 6 |
| Y-Kriegenbrunn 431/438 tripod (3 legs) | the 438/432 sister tripod is in service | 8 / 19 / 12 |
| Y-Landesbergen/Wechold/Sottrum rot/1 (220 kV, 3 legs) | Stade–Landesbergen 380 kV new-build corridor; gelb/2 in service | 18 / 12 / 19 |
| Meissen–Veltheim A09 | v5 2024-02-29 "Out of service"; A08 absent from r9 | 9 |

For this group I cannot tell whether any month of the window, or any of the v5 hours, falls inside an outage. The v4.1 extreme hours are 2025-11-03, 2026-01-22/29, 2026-04-30, 2026-06-29, 2026-08-13/23/30 and 2026-10-23. No published outage plan was found. Per the owner rules there are no operator requests, so this stays open.

Sensitivity option, if the owner wants it: run the v5 gate with this group switched off, as an N-k variant, to bound the effect. That is not done here.

## OSM status
Every circuit except the three with 0 km (Grohnde–Würgassen 2, Hradec–Etzenricht 441, Audorf T421) still has a mapped OSM route of the full length. OSM carries no disused tag on them, so OSM neither confirms nor contradicts an outage, and its status tags may be stale anyway.

## Sources
- TenneT Infoblatt "Generalsanierung Ottenhofen–Isar" (2024-03-07): https://vg-altfraunhofen.de/media//bilder/240307_Infoblatt_Generalsanierung_Ottenhofen-Isar.pdf
- TenneT notice Isar–Pleinting LH-08-B117 (2023-05-11): https://www.Vilshofen.de/fileadmin/Gemeinde/Rathaus/Bekanntmachungen/2023-05-11-Tennet_TSO_GmbH_380-kV-Leitung_Isar-Pleinting_LH-08-B117_Instandhaltungsmassnahme-_Erneuerung_der_Leiterseile_und_Isolatorenketten.pdf
- Bayernwerk/TenneT press release 2022-11 on the Raitersaich–Grafenrheinfeld line: https://bayernwerk-ag.mynewsdesk.com/pressreleases/bayernwerk-netz-und-tennet-finden-loesung-fuer-mehr-gruenstrom-3218852.pdf
- NEP TTG-P133 Borken–Gießen/Nord: https://www.netzentwicklungsplan.de/projektbibliothek/download/58795/TTG-P133%20Netzverst%C3%A4rkung%20zwischen%20Borken%20und%20Gie%C3%9Fen_Nord.pdf
- NEP P54 Irsching–Zolling–Ottenhofen: https://www.netzentwicklungsplan.de/projektbibliothek/download/58278/P54%20Netzverst%C3%A4rkung%20Irsching%20-%20Zolling%20-%20Ottenhofen.pdf

## EOWS tripod against "Doerpen West – Meppen EMSLD WB" (detection only, 2026-11-09)

The model holds both, and they are **two different circuits, not a duplicate**:
- **EMSLD WB** (11T0-0000-2116-Q) is the west circuit. r7 (2025-03-01) replaced Amprion's "Doerpen West – Hanekenfaehr EMSLD WB" (11T-D2-D7-00002F, "Removed") by "Doerpen West – Meppen EMSLD WB" ("New Entry"): a cut-in of the WB circuit at Meppen. In the same release, Hanekenfaehr–Meppen MEPPEN (0640-J) was replaced by BACCUM O/W.
- **EOWS / EMSLD OW** (0983-G, 0990-L, 0991-I) is the east circuit. It runs Meppen – Y-Niederlangen – Doerpen/W, with Amprion naming its share "Meppen – Niederlangen EMSLD OW". Its EIC differs from 2116-Q, and r7 does not touch it.
- So r7 does not supersede the tripod. The only open point for EOWS is its v6 2024-08-30 "Out-of-service." flag with no later entry. That is the same situation as the other 18 open circuits.

Proposals (not applied):
1. Keep the tripod in service and list it with the "open" group (JAO r6 flag without a later entry).
2. A separate value issue: 2116-Q carries X 9.698 Ω, the **sum** of Amprion's 4.849 Ω and TenneT's 4.8491 Ω. Both TSOs report the same whole-circuit R/X/B, the same pattern as C1 Bürs. The single value would be X 4.849 Ω, R 0.433 Ω, B 63.7 µS over the TenneT length of 32.96 km. That is a candidate C-class change for the owner.
3. 0983-G (Y-Meppen leg) carries estimated X 5.08 Ω. Amprion publishes 2.96 Ω and TenneT 2.88 Ω for the same leg, which looks like the same double-report pattern. To review together with item 2.
