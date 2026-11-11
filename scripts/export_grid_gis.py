"""Export the frozen grid model to a GeoPackage (and GeoJSON for the web viewer).

Usage::

    python scripts/export_grid_gis.py MODEL.pkl OUT_DIR [--web WEB_DIR] [--andes CASE.xlsx] \
        [--overlay OVERLAY.parquet] [--towers hv_lines.parquet [--towers-overpass X.json]] \
        [--stations osm/DEU.gpkg] [--diagrams] [--network v5/network] \
        [--station-table v5/stations] [--osm-substations osm/DEU.gpkg] [--edge-review CSV]
    python scripts/export_grid_gis.py --app apps/grid_map   # QGIS styles/project, symbols

``--app`` (re)writes the QGIS styles and project, the SVG symbols and ``style.js`` from
:mod:`tranpy.grid_gis_style`; it needs no model.
"""

from __future__ import annotations

import argparse

from tranpy.grid_gis import export_grid_gis
from tranpy.grid_gis_style import write_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", nargs="?", help="frozen model pickle (s, m)")
    parser.add_argument("out_dir", nargs="?", help="folder for grid_model.gpkg")
    parser.add_argument("--app", help="write QGIS styles/project and web symbols into this folder")
    parser.add_argument("--web", help="folder for the GeoJSON layers of the web viewer")
    parser.add_argument("--andes", help="ANDES .xlsx case of the model (simulation bus ids)")
    parser.add_argument("--overlay", help="geometry overlay parquet (corrected sites/routes)")
    parser.add_argument("--towers", help="OpenStreetMap line export (tower_lines layer)")
    parser.add_argument(
        "--towers-overpass", nargs="*", default=[], help="Overpass JSON ways added to --towers"
    )
    parser.add_argument(
        "--stations",
        help="station footprints: OpenStreetMap GeoPackage (power_substation_polygon) "
        "or a station table with site + polygon",
    )
    parser.add_argument(
        "--diagrams",
        action="store_true",
        help="write a pypowsybl single-line diagram per substation to WEB_DIR/sld",
    )
    parser.add_argument(
        "--network",
        help="grid_network output folder: draw AC circuits on their OSM way pieces",
    )
    parser.add_argument(
        "--osm-substations",
        help="OpenStreetMap GeoPackage (power_substation_polygon): other yards of a station",
    )
    parser.add_argument(
        "--station-table", help="station table folder (stations.parquet) for the station popup"
    )
    parser.add_argument(
        "--edge-review",
        help="CSV way_id,osm_n,model_n,reason: reviewed (explained) model/OSM differences",
    )
    parser.add_argument(
        "--way-notes",
        help="CSV way_id,note: reviewed notes (e.g. expansion project) on raw tower_lines ways",
    )
    parser.add_argument("--borders", help="Natural Earth countries GeoJSON (default: download)")
    args = parser.parse_args()
    if args.app:
        for key, value in write_all(args.app).items():
            print(key, len(value) if isinstance(value, list) else value)
    if args.model:
        if not args.out_dir:
            parser.error("OUT_DIR is required with MODEL")
        path = export_grid_gis(
            args.model,
            args.out_dir,
            borders_path=args.borders,
            andes_case=args.andes,
            web_dir=args.web,
            overlay_path=args.overlay,
            towers_path=args.towers,
            towers_overpass=tuple(args.towers_overpass),
            stations_path=args.stations,
            diagrams=args.diagrams,
            network_dir=args.network,
            station_table=args.station_table,
            osm_substations_path=args.osm_substations,
            edge_review=args.edge_review,
            way_notes=args.way_notes,
        )
        print(path)


if __name__ == "__main__":
    main()
