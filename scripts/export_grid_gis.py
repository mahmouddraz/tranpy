"""Export the frozen grid model to a GeoPackage (and GeoJSON for the web viewer).

Usage::

    python scripts/export_grid_gis.py MODEL.pkl OUT_DIR [--web WEB_DIR] [--andes CASE.xlsx] \
        [--overlay OVERLAY.parquet] [--towers hv_lines.parquet [--towers-overpass X.json]]
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
        )
        print(path)


if __name__ == "__main__":
    main()
