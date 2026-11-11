# Map test panel

Fixed places checked before and after every map build (plan: map on OSM topology). Each place
gets a PNG from the live map (`before/`) and from staging (`after/`), at the given centre and
zoom, viewport 1400×1000. The automatic checks are in `scripts/map_acceptance.py`.

- **owner**: the owner's screenshots and named cases (21; P62/P63 Marzahn and Gransee added after
  the staging review; Bentwisch box = P05, the 293/294/295 junction = P19). The 294/295/293 junction is the end
  shared by OSM 220 kV ways with refs 293, 294 and 295 (Nordring Berlin).
- **fragmented**: all 22 junctions with more than 2.5 drawn pieces per circuit within 1 km
  (the `junction_fragmented_drawing` scan of `issues/08-map-connections`; several share a point).
- **straight**: 10 of the 26 `straight_no_route` circuits, drawn at their midpoint, seeded sample.
- **random**: 10 German substations/tees not listed above, `numpy.random.default_rng(20261109)`.

Built from the v5_3e export by `~/data/grid/acceptance/tools/build_panel.py` on cairo-lab;
machine-readable copy: `~/data/grid/acceptance/panel.csv`. PNGs:
`~/data/grid/acceptance/before/<id>.png` (rendered with `tools/shoot_panel.py`, headless
Chromium, map URL `http://localhost:8080/`, view set with `window.gridMap.jumpTo`).

| id | kind | place | lon | lat | zoom | why |
|---|---|---|---|---|---|---|
| P01 | owner | Landesbergen | 9.12527 | 52.53827 | 14 | owner screenshot: station layout |
| P02 | owner | Vieselbach | 11.12254 | 50.99944 | 14 | owner screenshot: crowded station |
| P03 | owner | Mutterstadt | 8.35226 | 49.42123 | 14 | owner screenshot |
| P04 | owner | Mehrum / Mehrum-Nord | 10.08753 | 52.31566 | 15 | convergence point, Mehrum-Nord leads (model topology) |
| P05 | owner | Bentwisch box | 12.21531 | 54.10624 | 15 | 543/544 straight shortcut, 220 kV box |
| P06 | owner | Bentwisch 544 yard | 12.21636 | 54.10033 | 15 | 544 end at the 380 kV yard |
| P07 | owner | Siedenbrünzow | 13.14362 | 53.9077 | 14 | owner screenshot: lead way reused |
| P08 | owner | Marke | 12.26223 | 51.74112 | 13 | plant attachment straight lines |
| P09 | owner | Perleberg | 11.85244 | 53.08693 | 14 | owner screenshot |
| P10 | owner | Güstrow | 12.18688 | 53.81539 | 14 | 423/424, 543/544 straight leg |
| P11 | owner | Pasewalk | 14.03473 | 53.50745 | 14 | 315/316 straight shortcut |
| P12 | owner | Herbertingen | 9.42497 | 48.06451 | 14 | owner screenshot |
| P13 | owner | Remptendorf | 11.64569 | 50.5351 | 14 | owner screenshot; 463/464 end 136 m off |
| P14 | owner | Rommerskirchen | 6.70258 | 51.00874 | 14 | 21 station-end pieces |
| P15 | owner | Sechtem | 6.97267 | 50.79567 | 14 | station-end pieces |
| P16 | owner | Daxlanden | 8.31372 | 49.00918 | 14 | station-end pieces |
| P17 | owner | Kriftel | 8.47185 | 50.09734 | 14 | owner case |
| P18 | owner | Westtirol | 10.87391 | 47.244 | 13 | foreign end (AT) |
| P19 | owner | 294/295/293 junction (Nordring Berlin, …nhagen – Wustermark) | 13.2063 | 52.6586 | 15 | jumps and loose ends at junction |
| P20 | fragmented | y-kriftel#trebur s | 8.49022 | 50.0851 | 15 | 25 pieces of 6 circuits within 1 km |
| P21 | fragmented | y-wechold#gelb/2 | 9.13795 | 52.84703 | 15 | 22 pieces of 6 circuits within 1 km |
| P22 | fragmented | y-wechold#rot/1 | 9.13795 | 52.84703 | 15 | 22 pieces of 6 circuits within 1 km |
| P23 | fragmented | y-edelstahl#fichth o | 6.53964 | 51.297 | 15 | 18 pieces of 5 circuits within 1 km |
| P24 | fragmented | y-ried#ried o | 8.41896 | 49.71088 | 15 | 18 pieces of 5 circuits within 1 km |
| P25 | fragmented | y-ohligs#oerkhs o | 7.00393 | 51.17671 | 15 | 28 pieces of 8 circuits within 1 km |
| P26 | fragmented | y-ohligs#oerkhs w | 7.00393 | 51.17671 | 15 | 28 pieces of 8 circuits within 1 km |
| P27 | fragmented | y-waldlaubersheim#wonneg o | 7.81831 | 49.92269 | 15 | 17 pieces of 5 circuits within 1 km |
| P28 | fragmented | y-koblenz#soonwd o | 7.50519 | 50.34734 | 15 | 20 pieces of 6 circuits within 1 km |
| P29 | fragmented | y-krun#250 | 11.28694 | 47.49286 | 15 | 20 pieces of 6 circuits within 1 km |
| P30 | fragmented | y-krun#251 | 11.28694 | 47.49286 | 15 | 20 pieces of 6 circuits within 1 km |
| P31 | fragmented | y-osterath#frixhm n | 6.70995 | 51.08811 | 15 | 30 pieces of 9 circuits within 1 km |
| P32 | fragmented | y-kuhmoos#bl | 8.04518 | 47.55262 | 15 | 33 pieces of 10 circuits within 1 km |
| P33 | fragmented | y-gutersloh#gueter s | 7.99342 | 51.82154 | 15 | 12 pieces of 4 circuits within 1 km |
| P34 | fragmented | y-huntorf#gelb | 8.40602 | 53.23478 | 15 | 36 pieces of 12 circuits within 1 km |
| P35 | fragmented | y-huntorf#weiss | 8.40602 | 53.23478 | 15 | 36 pieces of 12 circuits within 1 km |
| P36 | fragmented | y-lippborg#geithe w | 7.98235 | 51.67718 | 15 | 37 pieces of 13 circuits within 1 km |
| P37 | fragmented | y-lippborg#haarst o | 7.98235 | 51.67718 | 15 | 37 pieces of 13 circuits within 1 km |
| P38 | fragmented | y-bellendorf#polsum s | 7.03681 | 51.60003 | 15 | 16 pieces of 6 circuits within 1 km |
| P39 | fragmented | y-osterath#toesee w | 6.71919 | 51.3413 | 15 | 16 pieces of 6 circuits within 1 km |
| P40 | fragmented | y-rheinhausen#stratm o | 6.72538 | 51.35768 | 15 | 23 pieces of 9 circuits within 1 km |
| P41 | fragmented | y-vohringen#donau o | 9.88535 | 48.3058 | 15 | 23 pieces of 9 circuits within 1 km |
| P42 | straight | Klixbuell/S - Endrup 2 | 8.78803 | 55.16278 | 12 | drawn straight, no matched route |
| P43 | straight | Doetinchem - Niederrhein DOETM WS | 6.69969 | 51.79578 | 12 | drawn straight, no matched route |
| P44 | straight | Y-Neurott (-GKM - Weinheim) rt | 8.56234 | 49.40095 | 12 | drawn straight, no matched route |
| P45 | straight | Y-Simbach (-St. Peter - Altheim) 233/230 | 13.03082 | 48.26503 | 12 | drawn straight, no matched route |
| P46 | straight | Gronau - Niederrhein PRHAMSBW | 6.8074 | 52.00107 | 12 | drawn straight, no matched route |
| P47 | straight | Y-Erzhausen (-Hardegsen - Lamspringe) 4 | 9.93723 | 51.89636 | 12 | drawn straight, no matched route |
| P48 | straight | Y-Brunsbuettel (-Brunsbuettel 50H - Hamburg/N 50H) 951 | 9.20074 | 53.89667 | 12 | drawn straight, no matched route |
| P49 | straight | Dausersberg - Eiserfeld SIEGRL O | 7.94589 | 50.8259 | 12 | drawn straight, no matched route |
| P50 | straight | Y-Simbach (-St. Peter - Altheim) 234/230 | 13.03095 | 48.26492 | 12 | drawn straight, no matched route |
| P51 | straight | Y-Y (-Buers - Herbertingen - Tiengen - Werben) BLUDNZ W | 9.63926 | 47.66697 | 12 | drawn straight, no matched route |
| P52 | random | asslar | 8.48073 | 50.59138 | 14 | random substation (seed 20261109) |
| P53 | random | bergrheinfeld/west | 10.15086 | 49.98212 | 14 | random substation (seed 20261109) |
| P54 | random | beuren | 8.862 | 47.8064 | 14 | random substation (seed 20261109) |
| P55 | random | kuhmoos | 7.95776 | 47.59598 | 14 | random substation (seed 20261109) |
| P56 | random | marbach | 9.23608 | 48.92687 | 14 | random substation (seed 20261109) |
| P57 | random | meitingen | 10.86753 | 48.53951 | 14 | random substation (seed 20261109) |
| P58 | random | ried | 8.42696 | 49.70761 | 14 | random substation (seed 20261109) |
| P59 | random | rohrsdorf | 12.81493 | 50.8577 | 14 | random substation (seed 20261109) |
| P60 | random | wuhlheide | 13.50634 | 52.47373 | 14 | random substation (seed 20261109) |
| P61 | random | y-bruchsal/kandelweg#rt | 8.60584 | 49.16169 | 14 | random tee (seed 20261109) |
| P62 | owner | Marzahn | 13.52432 | 52.53556 | 14 | owner (staging 8094): loop/zigzag at 921/922 and 495 |
| P63 | owner | Gransee | 13.26349 | 52.99413 | 14 | owner (staging 8094): zigzag |

## Acceptance checks

Run on every export before publishing:

```
python scripts/map_acceptance.py --export <grid_model.gpkg | web dir> \
    --model ~/data/grid/v5/model_v5_<x>.pkl --out ~/data/grid/acceptance/<build>
```

Hard (exit 1): drawn once, count equals model, ends on own box/node (25 m), no undashed straight
piece (2-vertex piece > 1 km or span > 3 km without `route_quality=straight`), no gap > 20 m,
slot changes only where the circuit set changes, metadata sha equals the model sha. Listed:
route/model length ratio outside [0.7, 1.5]. Baseline (v5_3e): `~/data/grid/acceptance/baseline_v5_3e/`.

Geometry checks (hard, added after the staging review of zigzags, loops and spikes):
`on_own_way` (drawn line within 30 m of its own OSM ways, plus the baked bundle offset
`(max_n-1)/2 x 90 m`; the strict 30 m share is reported too), `no_zigzag` (lateral jump > 40 m
between consecutive vertices, steeper than 45 deg to the way, or a switch between ways that do not
touch), `no_loop` (self-crossing, or a turn > 150 deg within 200 m), `no_spike` (out > 50 m from
the local median distance to the way and back within 150 m), `station_end` (more than 50 m of line
inside the circuit's own end box). Every finding is a point in `events.csv`.

## Visual check of all stations

`scripts/map_visual_check.py --url <served map> --export <web dir> --events <events.csv> --out <dir>`
renders every substation/tee/junction at zoom 15 (MapLibre in headless Chromium, 4 pages),
counts the findings within 1.5 km, and writes `places.csv`, `flags/<id>.csv`, `ranked.csv`,
4x4 contact sheets (`sheets/`, worst first) and `index.html`. Staging baseline before the rework:
`~/data/grid/acceptance/visual_staging_v1/`.
