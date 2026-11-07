#!/usr/bin/env bash
# Rebuild the QWC2 grid project from the latest export and regenerate the QWC config.
# Run on cairo-lab:  ~/data/qwc/publish.sh   (or from the repo: apps/grid_map/qwc/publish.sh)
# Source: the newest of ~/data/grid/web/grid_model_qgis.zip (GeoPackage + styles of
# one export) and ~/data/grid/grid_model.gpkg + ~/data/grid/qgis/styles.
# Env overrides: GRID_ZIP, GRID_GPKG, GRID_STYLES, QWC_DIR.
set -euo pipefail
QWC_DIR=${QWC_DIR:-$HOME/data/qwc/qwc-docker}
GRID_ZIP=${GRID_ZIP:-$HOME/data/grid/web/grid_model_qgis.zip}
GRID_GPKG=${GRID_GPKG:-$HOME/data/grid/grid_model.gpkg}
GRID_STYLES=${GRID_STYLES:-$HOME/data/grid/qgis/styles}
HERE=$(cd "$(dirname "$0")" && pwd)
RES=$QWC_DIR/volumes/qgs-resources
cd "$QWC_DIR"

if [ -f "$GRID_ZIP" ] && [ "$GRID_ZIP" -nt "$GRID_STYLES" ]; then
  EXP=$QWC_DIR/build/export
  rm -rf "$EXP" && mkdir -p "$EXP"
  python3 -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$GRID_ZIP" "$EXP"
  GRID_GPKG=$EXP/grid_model.gpkg
  GRID_STYLES=$EXP/styles
fi
echo "source: $GRID_GPKG + $GRID_STYLES"

# 1. Snapshot data + styles (copy, so a running re-export never half-overwrites the served file).
mkdir -p "$RES/grid/styles" "$QWC_DIR/build"
cp "$HERE/build_project.py" "$QWC_DIR/build/"
python3 -c "import sqlite3,sys; s=sqlite3.connect(sys.argv[1]); d=sqlite3.connect(sys.argv[2]); s.backup(d); d.execute('pragma journal_mode=delete'); d.close()" "$GRID_GPKG" "$RES/grid/grid_model.gpkg.new"
mv "$RES/grid/grid_model.gpkg.new" "$RES/grid/grid_model.gpkg"
rm -f "$RES/grid/styles/"*.qml
cp "$GRID_STYLES"/*.qml "$RES/grid/styles/"
[ -d "$GRID_STYLES/../symbols" ] && cp -r "$GRID_STYLES/../symbols" "$RES/grid/" || true

# 2. Build the .qgs with PyQGIS in a one-off qgis-server container.
docker compose -p tranpy-qwc run --rm --no-deps -T --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$RES:/work" -v "$QWC_DIR/build:/build:ro" --entrypoint python3 \
  qwc-qgis-server /build/build_project.py \
  /work/grid/grid_model.gpkg /work/grid/styles /work/grid_model.qgs

# 3. Regenerate the QWC service configs and the viewer themes.
docker compose -p tranpy-qwc exec -T qwc-api-gateway \
  curl -sf -X POST 'http://qwc-config-service:9090/generate_configs?tenant=default'
echo
echo "published: http://$(hostname):8088/"
