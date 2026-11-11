#!/usr/bin/env bash
# Promote a STAGING build of the grid map (8094) to PRODUCTION: the web map (8080) and the
# QWC/QGIS map (8088). Run on cairo-lab, only after the owner approved the staging build:
#
#     bash scripts/publish_map.sh --dry-run ~/data/grid/staging_osmgraph/web_v5_1_c16
#     bash scripts/publish_map.sh           ~/data/grid/staging_osmgraph/web_v5_1_c16
#     bash scripts/publish_map.sh --rollback          # restore the copies of the last publish
#
# Steps (see docs/grid_map.md, "Staging vs production"):
#   0. the staging build must name the frozen model sha ($MODEL_SHA);
#   1. scripts/make_prod_web.py writes the clean production folder $PROD_DIR (new folder,
#      default ~/data/grid/prod_build_<staging folder>): data without debug fields, app
#      files with the production profile, GeoPackage + clean QGIS styles + download zip;
#   2. the QWC project is built from that GeoPackage in the QGIS server container and loaded
#      back (qgis/qgs_check.json);
#   3. prod acceptance (scripts/prod_acceptance.py) against the staging source, with the app
#      served on 127.0.0.1:$TEST_PORT; any failure stops here;
#   --dry-run stops here and prints what the publish would do;
#   4. rollback copies (web, app files, qgis) tagged before_<prod build id>; the tag is kept
#      in ~/data/grid/ROLLBACK_LAST_PROD.txt;
#   5. publish: data -> ~/data/grid/web (8080 data), app files -> ~/data/grid/live_map,
#      qgis/ -> ~/data/grid/qgis, then apps/grid_map/qwc/publish.sh rebuilds 8088 from
#      web/grid_model_qgis.zip; 8080 must then report mode production.
set -euo pipefail
GRID=${GRID:-$HOME/data/grid}
REPO=${REPO:-$HOME/code/tranpy}
LIVE_APP=${LIVE_APP:-$GRID/live_map}
QWC_DIR=${QWC_DIR:-$HOME/data/qwc/qwc-docker}
PY=${PY:-$HOME/code/tranpy/.venv/bin/python}     # geopandas (GeoPackage)
PW_PY=${PW_PY:-$HOME/venvs/pw/bin/python}        # playwright (acceptance)
TEST_PORT=${TEST_PORT:-8099}
MODEL=${MODEL:-$GRID/v5/frozen_v5_1/model_v5_1_frozen.pkl}   # read only
MODEL_SHA=${MODEL_SHA:-cc4589c173f9466408d44b043105b5213c12fed17bc9ce2ad2f9a4cc922b723e}
APP_FILES=(app.js index.html style.css style.js symbols)
DRY=0
case "${1:-}" in
  --dry-run) DRY=1; shift ;;
  --rollback)
    TAG=${2:-$(cat "$GRID/ROLLBACK_LAST_PROD.txt")}
    test -d "$GRID/web_$TAG" && test -d "$GRID/app_$TAG"
    rsync -a --delete "$GRID/web_$TAG/" "$GRID/web/"
    [ -d "$GRID/qgis_$TAG" ] && rsync -a --delete "$GRID/qgis_$TAG/" "$GRID/qgis/"
    for f in "${APP_FILES[@]}"; do [ -e "$GRID/app_$TAG/$f" ] && rsync -a --delete "$GRID/app_$TAG/$f" "$LIVE_APP/"; done
    touch "$GRID/web/grid_model_qgis.zip"   # newest source for the QWC rebuild
    bash "$REPO/apps/grid_map/qwc/publish.sh"
    echo "rolled back 8080 and 8088 to the *_$TAG copies"
    exit 0 ;;
esac
STAGE=$(realpath "${1:?usage: publish_map.sh [--dry-run] <staging web build dir> | --rollback [tag]}")
PROD_DIR=${PROD_DIR:-$GRID/prod_build_$(basename "$STAGE")}

# 0. staging build of the frozen model
test -f "$STAGE/metadata.json" && test -f "$STAGE/lines.geojson" && test -d "$STAGE/sld"
grep -q "\"source_sha256\": \"$MODEL_SHA\"" "$STAGE/metadata.json" \
  || { echo "staging metadata does not name model sha $MODEL_SHA" >&2; exit 1; }

# 1. clean production folder (written once; an existing one must come from the same staging build)
if [ -d "$PROD_DIR" ]; then
  grep -q "\"staging_source\": \"$STAGE\"" "$PROD_DIR/data/metadata.json" \
    || { echo "$PROD_DIR exists but was built from another staging folder" >&2; exit 1; }
  echo "1. reusing $PROD_DIR"
else
  nice "$PY" "$REPO/scripts/make_prod_web.py" "$STAGE" "$PROD_DIR" \
    --app "$REPO/apps/grid_map" --styles "$REPO/apps/grid_map/qgis/styles" --model "$MODEL"
fi
BUILD=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["build_id"])' "$PROD_DIR/data/metadata.json")

# 2. QWC project from the clean GeoPackage, loaded back with PyQGIS (one-off container)
(cd "$QWC_DIR" && docker compose -p tranpy-qwc run --rm --no-deps -T --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PROD_DIR/qgis:/work" -v "$REPO/apps/grid_map/qwc:/build:ro" --entrypoint sh qwc-qgis-server -c \
  "python3 /build/build_project.py /work/grid_model.gpkg /work/styles /work/grid_model.qgs \
   && python3 /build/check_project.py /work/grid_model.qgs /work/qgs_check.json")

# 3. prod acceptance, app served on a test port (this script's own server only)
(cd "$PROD_DIR" && exec nice python3 -m http.server "$TEST_PORT" --bind 127.0.0.1 >/dev/null 2>&1) &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
sleep 2
nice "$PW_PY" "$REPO/scripts/prod_acceptance.py" "$STAGE" "$PROD_DIR" --url "http://127.0.0.1:$TEST_PORT/" \
  --model "$MODEL" --py "$PY"
kill $SRV 2>/dev/null || true

TAG=before_prod_$BUILD
run() { if [ $DRY = 1 ]; then echo "  would run: $*"; else "$@"; fi; }
echo "publish $BUILD (staging $STAGE) -> 8080 + 8088; rollback tag $TAG"

# 4. rollback copies of the current production
[ -e "$GRID/web_$TAG" ] || run rsync -a "$GRID/web/" "$GRID/web_$TAG/"
[ -e "$GRID/qgis_$TAG" ] || run rsync -a "$GRID/qgis/" "$GRID/qgis_$TAG/"
if [ ! -e "$GRID/app_$TAG" ]; then
  run mkdir -p "$GRID/app_$TAG"
  for f in "${APP_FILES[@]}"; do run rsync -a "$LIVE_APP/$f" "$GRID/app_$TAG/"; done
fi
run sh -c "echo $TAG > '$GRID/ROLLBACK_LAST_PROD.txt'"

# 5. publish: 8080 data and app, QGIS files, then QWC 8088 from the clean zip
run rsync -a --delete "$PROD_DIR/data/" "$GRID/web/"
for f in "${APP_FILES[@]}"; do run rsync -a --delete "$PROD_DIR/$f" "$LIVE_APP/"; done
run rsync -a --delete --exclude qgs_check.json --exclude grid_model.qgs "$PROD_DIR/qgis/" "$GRID/qgis/"
run touch "$GRID/web/grid_model_qgis.zip"
run bash "$REPO/apps/grid_map/qwc/publish.sh"
if [ $DRY = 1 ]; then echo "dry run: nothing published"; exit 0; fi
curl -sf "http://localhost:8080/data/metadata.json" | grep -q '"mode": "production"' \
  && echo "published $BUILD -> http://$(hostname):8080/ and :8088/ (rollback: $0 --rollback)"
