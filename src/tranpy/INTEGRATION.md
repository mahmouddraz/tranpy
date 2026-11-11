# grid_bundles: integration notes (for agent F)

Display only. Gives every drawn circuit piece an integer `slot` and `n` so the map draws
parallel circuits side by side with a **pixel** offset. Pieces stay exactly on their OSM
way geometry; no geometry is changed and no model file is read or written.

```python
import pandas as pd, geopandas as gpd, shapely
from tranpy.grid_bundles import read_network, piece_slots, route_slots

net = "~/data/grid/v5/network"
graph, routes = read_network(net)                          # edges/nodes/circuit_routes
pc = pd.read_parquet(f"{net}/circuit_pieces.parquet")
pieces = gpd.GeoDataFrame(pc.drop(columns="geometry"),
                          geometry=shapely.from_wkb(pc["geometry"].to_numpy()), crs="EPSG:4326")
pieces = piece_slots(graph, routes, pieces)                 # + slot, n (int)
```

Map (app.js): per piece,
`line-offset = (slot - (n - 1) / 2) * gap_px(zoom)` (MapLibre: positive = right of the
piece's drawn direction). `slot` already accounts for the direction each piece geometry is
drawn in, so do not re-orient pieces after this call. QGIS: the same with an offset in
map units or pixels.

Rules the slots follow:
- one circuit order per corridor (run of edges with the same circuit set), chosen by the
  metro-line crossing-minimisation pairwise ordering (see the module docstring); slots
  change only where the circuit set changes;
- pieces shorter than `MIN_PIECE_M` = 100 m take their neighbours' slot;
- circuits ending at a station are ordered by where they meet its outline;
- connectors (`edge_id` -1, A's short end to the box) and `join` pieces keep the slot of the piece next to
  them, so a line keeps its corridor slot up to the box; lines end where A's route
  touches the box.

`route_slots(graph, routes)` gives the same per route edge (`mrid`, `seq`, `edge_id`,
`forward`, `slot` in the edge's u→v frame, `n`) for `map_acceptance` (check: slot changes
only where the set changes).

Not used any more (still in the module, do not wire them in): `bundle_geometries`
(baked metre offsets), `station_fans` / `stations=` (schematic fans), `stub_from_export`.
With pixel offsets, consecutive pieces with different slots meet with a small step at the
node (MapLibre offsets each feature on its own); that is expected and only happens where
the circuit set changes.
