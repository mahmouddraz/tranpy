# QGIS project of the grid model

`grid_model.qgs` shows `grid_model.gpkg` (written by `scripts/export_grid_gis.py`) with the
map styles of `styles/`. The project uses relative paths, so the GeoPackage must sit in the
same folder as the project.

## Open it

1. Get the bundle (project, styles, GeoPackage):
   - on the tailnet: <http://cairo-lab:8080/data/grid_model_qgis.zip>, or
   - `scp -r cairo-lab:~/data/grid/qgis ./grid_qgis`.
2. Unzip, then open `grid_model.qgs` in QGIS 3.28 or newer (File > Open Project).
3. Switch the colouring per layer: right-click a layer > Styles > "By voltage" / "By TSO".
   The same styles are in `styles/<layer>_<mode>.qml` (Layer Properties > Style > Load Style).

If you open the GeoPackage on its own (drag it into QGIS), load the `.qml` files by hand.

## Notes

- The basemap is CARTO Positron (no labels), an XYZ layer: © OpenStreetMap contributors,
  © CARTO. It needs internet access; switch it off for offline work.
- Styles and project are written by `tranpy.grid_gis_style` (`python
  scripts/export_grid_gis.py --app apps/grid_map`). Change the specification there, not
  the XML, so the web viewer keeps the same colours and symbols.
- The XML was checked for well-formedness only; QGIS was not available on the export host,
  so the project has not been opened in QGIS by the author of the export.
