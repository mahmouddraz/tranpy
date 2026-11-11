# 01 Future assets: batch 2 (high-EIC circuits plus Lamspringe–Wahle and Borken–Mecklar)

Status: findings only. Nothing in `model_v4final` was changed. The rows are appended to `occurrences.csv` with `category` prefixed `batch2|`; earlier rows are kept.
Method: web search across TSO, NEP, state planning and trade sources, plus Overpass `[date:"2021-06-01T00:00:00Z"]` against the current state. Overpass rate-limited several queries (Sottrum, Mehringen, Dollern, Birkenfeld 2021). Those places rely on TSO sources only.

## Verdict counts (53 appended rows: 49 high-EIC, 2 Lamspringe–Wahle, 2 Borken–Mecklar)
| Verdict | Rows |
|---|---|
| before 2021 (Polsum cluster 6, Borken–Mecklar 2 likely) | 8 |
| after 2021, date with source | 8 (Birkenfeld 4, Lamspringe–Wahle 2, Röhrsdorf–Weida–Remptendorf 2) |
| after 2021 likely, no date | 6 (Mehringen 4, Sottrum–Stade/W 2) |
| unknown | 31 |

## Verified rows proposed for v5
| Circuits (EIC, fault_line, events/unstable) | Evidence | 2021 replacement |
|---|---|---|
| Birkenfeld – Pulverdingen ge 2488 (757, 18/7); Y-Birkenfeld tee 2490/2491/2492 (814, 815, 837; 9/0, 7/0, 13/0) | RP Karlsruhe 2022 plan approval "Neubau einer 380-kV-Leitung Birkenfeld – Punkt Ötisheim" (rpk.baden-wuerttemberg.de, LA 7620); Omexom 23 Feb 2026: "successfully completed and commissioned the new 380 kV transmission line" (https://www.omexom.com/?p=8229) | In 2021 Birkenfeld was 220 kV only (OSM name "220 kV-Umspannwerk Birkenfeld"). Remove the Birkenfeld 380 kV branch, then restore the untapped Philippsburg – Bruchsal/Kändelweg and Philippsburg – Pulverdingen 380 kV circuits. Take the old EICs from the JAO 2021 static grid model. |
| Lamspringe – Wahle 3/4 1694, 1695 (609, 610; 5/0, 9/1) | ZfK, 28 Nov 2024: "2022 die nördlichen Abschnitte in der Nord-Süd-Linie Peine-Salzgitter-Einbeck-Northeim in Betrieb genommen" (https://www.zfk.de/energie/strom/laengste-drehstromleitung-deutschlands-geht-ans-netz). Section A Wahle–Lamspringe plan approval 31.05.2019. | Remove the 380 kV circuits and restore the old 220 kV Wahle – Lamspringe route (EIC from JAO 2021). |
| Röhrsdorf – Weida 574 2251 (103, 10/1); Weida – Remptendorf 576 2252 (131, 11/1) | Global Transmission: west section Weida–Remptendorf partially commissioned 28 Mar 2024, east section lot 1 finished Feb 2024 (https://globaltransmission.info/german-50hertz-partially-commissioned-west-section-of-rohrsdorf-remptendorf-line/) | The new line replaces the old one in the same corridor. In 2021 the old 380 kV Röhrsdorf–Remptendorf circuits existed, so the topology stays. Restore the old EIC and conductor parameters. Whether Weida was tapped in 2021 is still open. |

Events on the proposed fault lines: 8 circuits, 93 events, 10 unstable. Birkenfeld–Pulverdingen alone accounts for 18 events and 7 unstable.

## After 2021 likely, not proposed (no commissioning date)
- Mehringen cluster: Landesbergen – Mehringen 3 (2295), Mehringen – Dollern grün-4/schwarz-3 (2470, 2471) and Y-Mehringen (2296); 45 events, 0 unstable. Tractebel describes Mehringen as a new 380 kV substation of the Stade–Landesbergen project (period 2020–2025). In 2021 this corridor was 220 kV Stade/West – Sottrum – Wechold – Landesbergen.
- Sottrum – Stade/W schwarz/weiss (2306, 2464); 25 events, 1 unstable. Same project. The Stade-area section was plan-approved 27.04.2018, but no energisation date was found. In OSM, a way named "Stade - Landsbergen" appears only in the current data, while a 380 kV substation at Stade-West already existed in 2021.
- If confirmed, the 2021 replacement is to remove these circuits and restore the 220 kV chain.

## Before 2021, keep
- Polsum cluster (2479, 2481, 2482, 2483, 2484, 2516): OSM on 2021-06-01 already shows "Station Polsum" at 380 kV with the 380 kV Kusenhorst, Lippe Süd and Scholven F circuits. The high EIC therefore reflects a relabel or bay reconfiguration, not a new line. Events: 66, unstable: 8. NEP M970 (Polsum–Niederrhein reconductoring, 2035) comes later.
- Borken – Mecklar 1/2 (11TD2L…): the EIC is old-style, the uprating notice treats the line as an existing 380 kV line, and the new Wahle–Mecklar Hessian section (2023) ends at Mecklar. Rated "likely"; no date was found.

## Unknown (31 rows)
These rows have no source in this batch: BACCUM O 2249, Bandstahl, Buer, the Fichtenhain/Utfort 220 kV cluster, Itterbach, Osterath – Rommerskirchen KNESTE O (12/6; the Ultranet conversion targets 2027, so the AC circuit status is unclear), Nassau tee, Ried tee, Soonwald tee, Trebur tee and Zeilsheim (19/3), Conneforde/O and Elsfleth/W (21/6), Etzenricht – Mechlenreuth, Helmstedt/Wahle tee, Eichstetten C1/C2, and the two out-of-service Rheingraben circuits. The next step is a JAO static grid model release-notice diff of 2021 against 2023–2025 for these EICs.

## Addendum: OSM 2021-06-01 results (overpass.kumi.systems mirror, run after the first report)
- Weida: in 2021 OSM already shows a 380 kV Weida substation with circuits "573 Remptendorf - Weida, 575 Weida - Röhrsdorf", alongside 220 kV 207/208/231/232. Weida was therefore tapped in 2021. The 2021 replacement for 2251/2252 (574/576) is to keep the topology and relabel the circuits to the old 573/575 (old EICs from JAO 2021), with the old conductor parameters.
- Sottrum: in 2021 OSM already shows 380 kV "3100 Dollern - Sottrum" and "3003 Sottrum - Landsbergen" at the 380/220/110 kV Sottrum substation, plus 220 kV "Abbenfleth-Sottrum" and Leitung 2144. A 380 kV Stade/W–Sottrum way does not appear. This supports "after 2021" for 2306/2464 (the 2021 link was presumably the 220 kV Abbenfleth/Stade–Sottrum), and it shows that the 2021 Dollern–Sottrum–Landesbergen 380 kV path ran without Mehringen.
- Mehringen, Birkenfeld and Lamspringe: the mirror returned no elements for 2021. Absence is not proof.
