# 12 Status of the 18 "open" circuits (issue 10) and the Erzhausen / Kühmoos gaps (KNOWN_LIMITS A2, A3b)

Analysis only: no model, map or PLAN.md changes were made, and nothing was committed. Proposals go to the owner, and approved ones go through the data agent's gate. The model is `model_v5_5q.pkl`. The v5 hours are `~/data/v5_hours/selected_hours.csv` (100 hours, Nov 2025–Oct 2026). Each item has one row in `occurrences.csv`, which gives every in-window outage interval, the hours out and the affected v5 hours.

## Method (ENTSO-E A78)
- **Internal DE lines are not served under internal domain pairs.** A78 with In = Out = the TenneT, 50Hertz, Amprion or TransnetBW control area (or DE-LU) returns an empty zip. DE–DE control-area pairs return "No matching data".
- **TenneT publishes its internal 380/220 kV elements under its border pairs.** It does this for every element that affects cross-zonal capacity. I queried A78 for CZ→TenneT (2024-06..2026-10, from issue 11) and for TenneT↔AT, NL, DK plus TenneT→CZ (2025-09..2026-10). That is about 10,000 documents in total. A CH↔TransnetBW query for the Kühmoos check was abandoned after one month, because the endpoint returned 599 errors.
- **Processing.** I kept the latest revision of each document and dropped those with status A09 (cancelled). All matched outages are A53 *planned*, reason B19 (maintenance), apart from one A54 forced outage on Landesbergen–Sottrum GE/2.
- **What `quantity` means.** The quantity in these documents is the border capacity during the outage, not the line rating. The asset itself is unavailable.
- **Storage.** The raw XML stays in the session scratchpad (`e2/`, `e3/`) and is not in the repo.
- **What absence means.** If a circuit is missing from these documents, nothing is published for it. That does not show the circuit is in service.

## Counts
| verdict | n | items |
|---|---|---|
| in service (short planned outages only) | 6 | Irsching–Ottenhofen 421; Kummerfeld–Dollern grün; Stadorf–Lüneburg 1; Wechold rot/1 tripod (3 legs) |
| out for part of the window | 3 | Isar–Ottenhofen 443; Isar–Pleinting 451; Borken–Gießen/N 2 |
| out for the whole window | 0 | – |
| unresolved (not published) | 9 | Bergrheinfeld/W–Grafenrheinfeld 426; Grafenrheinfeld–Raitersaich 433; Kriegenbrunn 431/438 tripod (3 legs); Klein Ilsede–Mehrum/N 1; Hamburg/N–Lübeck gelb; Voslapp–Inhausen rot; Meissen–Veltheim A09 |
| A2 Erzhausen / A3b Kühmoos | 2 | both: keep as they are (see below) |

The ENTSO-E data also explains the JAO v6 "Out-of-service" flag (2024-08-30) for the circuits it covers. In each case a long planned outage was running at the v6 reference date:
- 443: 2024-05-21..2025-05-15
- 451: 2024-08-26..2024-12-06
- 421 and Borken–Gießen 2: daily outages Jun–Oct 2024

All of these circuits were back in service afterwards. This supports issue 10's reading that the v6 batch was a snapshot of temporary outages.

## 1 Circuits out for part of the window (proposal: keep `in_operation` and switch off per hour)
The v5 hours are in UTC. Hours are 2026 unless marked 2025.

| circuit | in-window outages (A53/B19) | hours out / 8760 | v5 hours inside an outage |
|---|---|---|---|
| Isar–Ottenhofen 443 (11TD2L000000123B) | 2025-12-01; 2026-01-19..22 (days); **2026-05-11..07-21 continuous**; then Mon–Fri 05–16 UTC every week from 07-22 to 10-21 (09-28..10-09 continuous) | 3,227 (36.8 %) | 31: 01-22T07, 05-19T02, 05-24T15, 05-25T14, 06-01T03, 06-03T19, 06-10T10, 06-12T22, 06-14T18, 06-21T06, 06-27T20, 06-29T08 (summer_peak), 07-01T12, 07-03T10, 07-06T20, 07-19T04, 07-22T05, 07-31T09, 08-05T09, 08-06T11, 08-13T11 (high_solar), 08-26T18, 08-27T20, 08-31T05, 09-22T09, 09-30T05, 10-04T11, 10-05T15, 10-09T05, 10-10T08, 10-14T14. 01-22T07 is winter_peak |
| Isar–Pleinting 451 (11TD2L0000001265) | daytime outages 2025-11-17..12-01, 2026-01-26..03-20 (several whole weeks 02-02..02-20); **2026-03-25 11:00..05-07 15:00 continuous** | 1,713 (19.6 %) | 22: 2025-11-18T15, 01-29T10 (high_load), 02-09T11, 02-10T16, 02-28T06, 03-04T14, 03-12T07, 03-13T08, 03-26T18, 03-27T04, 04-02T01, 04-08T18, 04-10T08, 04-13T15, 04-17T00, 04-19T16, 04-20T02, 04-23T22, 04-28T17, 04-29T08, 04-30T16 (median_load), 05-02T04 |
| Borken–Gießen/N 2 (11TD2L0000000293) | daytime outages on 2025-11-04..12 and 2026-02-09..04-10; **2026-04-13..07-03 17:00 continuous**; 07-08, 08-01, 10-01 | 2,388 (27.3 %) | 29: 2025-11-06T13, 02-09T11, 03-04T14, 03-05T15, 03-12T07, 04-10T08, 04-13T15, 04-17T00, 04-19T16, 04-20T02, 04-23T22, 04-28T17, 04-29T08, 04-30T16 (median_load), 05-02T04, 05-09T16, 05-19T02, 05-24T15, 05-25T14, 06-01T03, 06-03T19, 06-10T10, 06-12T22, 06-14T18, 06-21T06, 06-27T20, 06-29T08 (summer_peak), 07-01T12, 07-03T10 |

Several outages overlap. Isar–Ottenhofen 443 and Borken–Gießen 2 are both out in 13 of the same v5 hours (May–July), and 451 and Borken–Gießen 2 share 13 hours (March–May). In total, 56 of the 100 v5 hours have at least one of the three circuits out. These are all planned outages, published ahead and still active (not cancelled) in their latest revision. A planned outage can still be shortened when the work is actually done, so these are confidence "high" for "planned out", not metered.

The background for each work comes from issue 10's sources:
- 443: TenneT "Generalsanierung Ottenhofen–Isar", Infoblatt 2024-03-07.
- 451: TenneT LH-08-B117 conductor renewal, 2023-05-11.
- Borken–Gießen/N 2: NEP TTG-P133 re-conductoring, 73 km, "Geplante Inbetriebnahme: 2027". https://www.netzentwicklungsplan.de/projektbibliothek/download/56204/TTG-P133%20Netzverst%C3%A4rkung%20zwischen%20Borken%20und%20Gie%C3%9Fen_Nord.pdf

The pattern in the 2026 outages, a continuous block followed by weekly daytime releases, is typical of conductor-replacement campaigns. I found no published completion date for any of the three beyond the NEP 2027 date for P133.

**Proposal.** Keep all three `in_operation` and clear `review`. Add an hour-dependent outage set (EIC → list of v5 hours above) that the case builder applies per hour. This does not change the base model.

## 2 In service
| circuit | ENTSO-E evidence | v5 hours affected |
|---|---|---|
| Irsching–Ottenhofen 421 | 117 docs; in window only 2026-07-27..30 (daytime) | none |
| Kummerfeld–Dollern grün (`TTG/380/DOLLERN-KUMMERFELD/GR`, same EIC) | 2026-01-22, 01-23, 02-17, 09-25 (8–9 h each) | 2026-01-22T07 (winter_peak), outage 07–15 UTC |
| Stadorf–Lüneburg 1 (`TTG/380/LUENEBURG-STADORF/1`, same EIC) | 2026-07-13 (6 h), 08-17 (8 h) | none |
| Wechold rot/1 tripod, 3 legs | published as a two-ended line, `TTG/220/LANDESBERGEN-SOTTRUM/RT1`, with a **new EIC 11T0-0000-2557-O**. GE/2 also has a new EIC (11T0-0000-2556-R). Outages: 2026-08-18..20 (days) and 09-29 (4 h) | none |

**Proposal.** Clear `review` on all six and record the ENTSO-E documents in `status_evidence`.

For the Wechold tripod, the change to a two-ended name suggests the Wechold tee is no longer a separate node. The model's three legs are equivalent in flow terms if the Wechold 220 kV node carries no load or generation. That check is a separate, minor item, and I propose no topology change now (confidence medium).

## 3 Unresolved (9 circuits)
None of these appears in any TenneT border publication. That means TenneT judges them irrelevant for cross-zonal capacity, so their absence says nothing about their state.

The web search found:
- 433 Grafenrheinfeld–Raitersaich (B114): maintenance since 2022-09. Bayernwerk/TenneT press release: https://bayernwerk-ag.mynewsdesk.com/pressreleases/bayernwerk-netz-und-tennet-finden-loesung-fuer-mehr-gruenstrom-3218852.pdf
- 433, lot 2: C-Team reference "Erneuerung von Isolatoren und Leiterseilen zwischen den Masten 91 und 189", undated: https://cteam.de/en/reference/raitersaich-grafenrheinfeld
- Kriegenbrunn: NEP P343 plans the full loop-in of Kriegenbrunn into Würgau–Raitersaich/West. "Today only single loop-in" means the tripod is the current state: https://www.netzentwicklungsplan.de/projektbibliothek/download/58135/P343%20Umspannwerkseinschleifungen%20TenneT.pdf
- None of these sources dates an outage in the window.

**Proposal.** None. Keep `in_operation` with the `review` flag. The owner's satellite list is in `satellite_checks.md`; it can only detect physical removal or construction, not whether a line is switched on. If the owner wants a bound, use the N-k sensitivity proposed in issue 10.

## 4 A2 Erzhausen
In the model:
- The 380 kV main legs (Y-Hardegsen 4/5 and Y-Lamspringe 4/5, published JAO values) meet at `y-erzhausen#4|380` / `#5|380`, so Hardegsen–Lamspringe flow is modelled correctly.
- The Erzhausen leg #4 (11T0-0000-2029-I) is a 0.27 km 220 kV placeholder with X 0.05 Ω, running to `y-erzhausen#4|220`. It is a dead stub.
- Junction #5 has no Erzhausen leg.
- The pumped-storage plant (200 MW, MaStR) is attached to `erzhausen|220`. That node is fed by the 220 kV Y-Erzhausen 1/3 leg towards Lehrte, which is 61 km long.

Evidence:
- **TenneT/ENTSO-E.** Since January 2026 (first document created 2026-01-20) TenneT names both 380 kV circuits `TTG/380/HARDEGSEN-ERZHAUSEN-LAMSPRINGE/4` (11T0-0000-1696-C) and `/5` (11T0-0000-1697-9). So at 380 kV Erzhausen is now a terminal of the circuits, a switching point. Planned outages: /4 on 2026-01-21, 09-28..10-09 and 10-17/18/27; /5 on 01-21, 09-21..25, 09-28..10-01, 10-17/18 and 10-28. Affected v5 hours: 2026-09-22T09 (/5), 09-30T05 (both) and 10-09T05 (/4).
- **Wahle–Mecklar** has been fully in operation since 2024-11-27: "Mit der Fertigstellung des verbleibenden Abschnitts im Großraum Göttingen ist die Gesamtleitung nun komplett in Betrieb" (ZfK, 2024-11-28, https://www.zfk.de/energie/strom/laengste-drehstromleitung-deutschlands-geht-ans-netz).
- **Plant connection still in planning in Dec 2023.** NLStBV UVP preliminary assessment of 2023-12-18, for the addition to the 8th plan change of Section B: "Verlegung der Erdkabel zwischen dem Ersatzneubau der Schaltanlage und dem Pumpspeicherwerk … in Kabelschutzrohren". It also says works on the cable "sowie zum Umbau bzw. der Erweiterung der Schaltanlage Erzhausen" must not overlap. https://uvp.niedersachsen.de/documents-ige-ng/igc_ni/8737d3da-5685-44a3-954d-60047642f087/231218_UVP-VP%20%28F%29.pdf
- **OSM** (`~/data/grid/osm/DEU.gpkg`):
  - The 380 kV spur 1305755114 (0.27 km, 2 circuits) has no construction tag.
  - The two 380 kV underground cables to the plant, 953739000 and 1457123443 (1.63 / 1.64 km), **are tagged construction**.
  - The 220 kV lines 296751942 ("Leitung 2001", 12.8 km) and 28215364 (5.2 km) are still mapped.
  - The 2017 plan approval lists LH-10-2001 (Godenau–Hardegsen) and LH-10-2013 (Abzweig Erzhausen) for dismantling.
- I found no Statkraft, TenneT or BNetzA statement that the plant has been switched to 380 kV.

**Verdict: keep as is** (confidence medium). The 380 kV switching point at Erzhausen appears to be in service, but nothing shows the plant connected at 380 kV. The 380 kV spur ends at a switchyard that has no load, generation or transformer in the model, so a dead stub is the correct power-flow representation. Its parameters do not matter: putting in typical 380 kV values over 0.27 km would change no flow.

If the owner's satellite check shows the cable trench closed and the 220 kV Abzweig Erzhausen dismantled, a later change would do the following, as one topology step (not proposed now):
- Move the plant to a new `erzhausen|380` node.
- Connect that node to `y-erzhausen#4|380` with 0.27 km of overhead line plus about 1.6 km of cable.
- Mark the change estimated.

## 5 A3b Kühmoos (y-kuhmoos#bl)
- **Swissgrid publishes no circuit-level grid data.** The ENTSO-E CH→TransnetBW A78 data lists only `220 kV Kühmoos-Laufenburg (Hotzenwald)` (10T-CH-DE-000123, the "ws" tie in the model). Nothing is published for 10T-CH-DE-000115.
- **Overpass (CH side, 7.85–8.25 E / 47.45–47.65 N).** It shows Laufenburg 380/220 kV (way 50319857, Swissgrid), several Swissgrid 220 kV routes from Laufenburg (for example 402053362, 29 km to the south, and 240692413/416 to the west) and Netzknoten Kühmoos. No substation in the area is named Lindenholz, and no source gives the junction position or the leg lengths. The only mention found is a third-party route listing, "Laufenburg – Lindenholz" (https://routes.tracestrack.com/route/6031722), which is not authoritative.
- **Verdict: keep as is.** There is no basis for parameters without guessing the topology on the Swiss side.
- **Effect on German flows: small.** Kühmoos keeps four modelled ties to Laufenburg (3 × 380 kV and 1 × 220 kV ws) plus the Tiengen tie. The DE–CH exchange is split across those ties, so the bl circuit's share, about one 220 kV circuit's worth, moves onto the parallel Kühmoos–Laufenburg ties a few km away. Internal German flows north of Kühmoos do not change.

## Proposed model changes (for the owner; approved ones go through the data agent's gate)
1. **Hour-dependent outage set for 443, 451 and Borken–Gießen/N 2.** Switch each off in the v5 hours listed in §1 (31 + 22 + 29 circuit-hours). Optionally also Kummerfeld–Dollern grün at 2026-01-22T07, and Hardegsen–Erzhausen–Lamspringe 4/5 at 09-22T09, 09-30T05 and 10-09T05 (/4: 09-30T05 and 10-09T05; /5: 09-22T09 and 09-30T05). This is not a static status change.
2. **Clear `review`** on the 9 resolved circuits (6 in service, 3 partial) and record the ENTSO-E A78 evidence in `status_evidence`.
3. **No change** for Erzhausen, Kühmoos or the 9 unresolved circuits.

## 6 Implementation 2026-11-09 (owner approved; data agent)
- **Review flags cleared** on the 9 resolved circuits (6 in service, 3 out for part of the window): `occurrences.csv` column `review` records the ENTSO-E A78 evidence for each. The 9 unresolved circuits keep the flag and are listed in KNOWN_LIMITS as "kept in service". The base model pickle is unchanged (3393038e).
- **Hourly outage set** `data/entsoe_a78_outages_v5.csv` (hour, time, circuit = model mrid, name, eic, a78_mrid, business_type, reason, start, end), built by `scripts/a78_outages.py` from all A78 documents fetched for the German control-area borders: TenneT with AT/NL/DK/CZ, 50Hertz with CZ/PL, Amprion with NL/BE/FR/LU/CH/AT, TransnetBW with FR/CH/AT (Nov 2025–Oct 2026; LU→Amprion has no data). 14,693 asset-documents (latest revision, A09 dropped), 596 assets, 266 model circuits of 220/380 kV, 2,522 circuit-hours over all 100 v5 hours (2–61 per hour; A53 2,489, A54 33). 443: 31 h, 451: 22 h, Borken–Gießen/N 2: 29 h (as in §1).
- The case builder applies it per hour (`grid_case.hour_outages` / `with_outages`, go4 `outages=`).
- **Hard failure (not worked around).** With all published outages, 33 of the 100 hours leave machines outside the main island (up to 48 machines, islands 9 → 16). Each case is a station whose two (or more) modelled 220/380 kV connections are published out at the same time: Hagenwerder (553+554, 14 h), Heinersdorf (548-4+444, 12 h), Eisenhüttenstadt (547-3+548-3, 11 h), Freiberg/Nord (592+594, 4 h), Stalldorf (416+Kupferzell rot, 7 h), Jardelund/Handewitt–Kassø (6 h), Brunsbüttel/Süderdonn (1 h, 1,174 MW), Teufelsbruch/Reuter (1 h). See `hourly_outage_islands.csv`. 443, 451 and Borken–Gießen/N 2 cause none of them. In reality these stations stay fed (busbar couplers, 110 kV or circuits not in the model, or the published element is one bay of a double circuit), so the owner must decide how such hours are treated.
