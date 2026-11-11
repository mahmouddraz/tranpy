# 13 Hourly outages: deferred, research note

Status: **not applied in v5** (owner decision 2026-11-09). v5 hours use the full grid without
planned maintenance (KNOWN_LIMITS). Code and data stay in the repo: `scripts/a78_outages.py`,
`data/entsoe_a78_outages_v5.csv`, `grid_case.hour_outages` / `with_outages` (off by default).

## What was tried (CHANGELOG 43)
Switching off every published ENTSO-E A78 outage of German 220/380 kV circuits per hour
(2,522 circuit-hours, median 25 circuits per hour, 266 circuits):
- 33 of 100 hours island stations, with machines cut off (Hagenwerder 14 h, Heinersdorf 12 h,
  Eisenhüttenstadt 11 h, Stalldorf 7 h, …): hard gate FAIL;
- overloaded rows 31 → 227, max loading 1.49 → 2.39, vmin 0.918 → 0.863.

## Why the simple approach is wrong
A planned outage never happens alone. The operator plans it together with remedial actions, and
the model has none of them:
1. **Redispatch / countertrading**: generation is moved so that no line is overloaded (n-1 secure).
2. **Topology changes**: busbar coupling, splitting, and switching of parallel circuits.
3. **Station internals**: double busbars and couplers, plus 110 kV support (not modelled), keep a
   station fed when "all" its modelled circuits are listed out. Often the A78 element is one
   bay or one system of a double circuit, not the whole connection.
4. **Dispatch consistency**: the hour's measured generation and flows already *include* the
   outage and the remedial actions. Applying the outage to a dispatch built without them gives
   a state that never happened.

## Approach to research
1. **Element mapping**: map each A78 asset (EIC/mRID) to the exact modelled element. Is it a whole
   circuit, one system, or a transformer/bay? Drop outages of elements the model does not
   represent separately.
2. **Station feasibility**: before removing, check whether the station keeps a path through
   couplers, a parallel system or 110 kV. Where the model cannot represent that, keep the
   element in and flag the hour.
3. **Remedial actions**: after applying the outages, run an n-1-aware OPF (pandapower OPF)
   that redispatches at minimum change from the hour's dispatch. Validate it against the
   *published* redispatch of that hour (which plants, direction, MW).
4. **Gate**: accept the hour only if the published redispatch is reproduced within a tolerance.
   Otherwise the hour stays without outages.
5. **Effect on TSA**: sensitivity of labels and CCT on a subset of hours, comparing with and
   without the outages. Start with the long-running circuits 443, 451 and Borken–Gießen/N 2.

## Data needed and where to get it (public, no operator requests)
| need | source | public? |
|---|---|---|
| planned/forced outages per element and hour | ENTSO-E A78 (transmission unavailability); already fetched | yes (research use) |
| redispatch per measure: plant, TSO, direction, MW, time, reason, affected element | netztransparenz.de "Redispatch" (§13 EnWG, all four TSOs); ENTSO-E costly remedial actions / redispatching (A63) and countertrading (A91) | yes |
| critical network elements, contingencies and outages per hour (Core flow-based) | JAO Publication Tool: CNEC list with Fmax/RAM per MTU; outages are reflected as contingencies/unavailable elements | yes |
| network element status/topology per hour (CGMES IGM/CGM) | ENTSO-E OPDE/CGM | **no** (TSO-only); the JAO static grid model is the closest public proxy |
| station layout (busbars, couplers, bays) | OSM `power=busbar/bay`, JAO static grid model, TSO single-line drawings where published | partly |
| 110 kV support around isolated stations | OSM 110 kV lines; DSO/TSO network maps | partly (geometry yes, parameters estimated) |
| generator availability (the other half of maintenance) | ENTSO-E A80/A77 (generation unavailability) | yes |

## Open questions for the research
- Do labels/CCTs change enough to matter for the paper's conclusions (sensitivity first)?
- Can the published redispatch be matched to model nodes reliably (plant → bus mapping)?
- Is a Core CNEC-based check (JAO) a better gate than our own overload count?

Not before the paper results on frozen v5; outage-aware hours are a candidate v6 item.
