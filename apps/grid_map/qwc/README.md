# German grid web GIS (QWC2 on cairo-lab)

The grid model (GeoPackage plus the QGIS styles from `tranpy.grid_gis` /
`tranpy.grid_gis_style`) is served as a QWC2 web GIS from cairo-lab. You only
need a browser on the Tailscale network:

    http://cairo-lab:8088/

It offers two map themes ("By voltage", "By TSO"; switch them with the eye icon
in Layers & Legend), identify on click, search for substations, lines, plants and
buses (including ANDES ids in the result titles), measuring, and printing (A4 and
A3 landscape PDFs with title, legend, scale bar, north arrow and attribution).

## Layout on cairo-lab

Everything lives under `~/data/qwc/`:

| Path | Content |
|---|---|
| `qwc-docker/` | clone of https://github.com/qwc-services/qwc-docker (the qgis/qwc-docker link redirects there) |
| `qwc-docker/docker-compose.yml` | the stack, compose project `tranpy-qwc` (copy: `stack/docker-compose.yml`) |
| `qwc-docker/.env` | `JWT_SECRET_KEY`, `POSTGRES_PASSWORD` (generated, mode 600, not in the repo) |
| `qwc-docker/api-gateway/nginx.conf` | gateway; re-resolves container names (copy: `stack/nginx.conf`) |
| `qwc-docker/volumes/config-in/default/` | `tenantConfig.json`, `themesConfig.json`, `config.json` (copies: `stack/config-in/default/`) |
| `qwc-docker/volumes/qgs-resources/` | published `grid_model.qgs` plus a snapshot `grid/grid_model.gpkg` and `grid/styles/*.qml` |
| `qwc-docker/volumes/config/` | configs written by the config generator |
| `qwc-docker/volumes/db/` | config database (Postgres) |
| `bin/` | `publish.sh` and `build_project.py` (copies of this directory) |

Images: `sourcepole/qwc-qgis-server:3.44` (QGIS 3.44 LTR), the
`sourcepole/qwc-*:latest-2026-lts` service images and `sourcepole/qwc-base-db:18`,
as in the qwc-docker example compose file; `nginx:1.29` as gateway.

Services: api-gateway (nginx), qgis-server, config-service (config generator),
ogc, feature-info, legend, print, permalink, auth (db-auth), map-viewer, postgis
(config DB only, no host port) and the one-shot config-db-migrate. Left out of the
example: admin GUI, data/editing, document, elevation, mapinfo, Solr and fulltext
search. Search uses the QWC2 `qgis` provider (WMS GetFeatureInfo with FILTER), so
it needs no search index. Permissions: `permissions_default_allow`, no users.

Ports: only `0.0.0.0:8088` (gateway). Network: `tranpy-qwc-net`. No named
Docker volumes (bind mounts under `~/data/qwc/qwc-docker/volumes`). Every
service has `restart: unless-stopped`, so the stack comes back after a reboot.
Resource limits: QGIS Server 4 CPUs, 4 GB and 4 FastCGI processes; the other
services 0.5 to 1 CPU each; no GPU.

## Start, stop, logs

    cd ~/data/qwc/qwc-docker
    docker compose -p tranpy-qwc up -d        # start / apply compose changes
    docker compose -p tranpy-qwc down         # stop and remove the containers (data stays)
    docker compose -p tranpy-qwc ps
    docker compose -p tranpy-qwc logs -f qwc-qgis-server

## Republish after a re-export

    ~/data/qwc/bin/publish.sh

The script takes the newest export: `~/data/grid/web/grid_model_qgis.zip`
(GeoPackage and styles of one export) if it is newer than
`~/data/grid/qgis/styles`, else `~/data/grid/grid_model.gpkg` with
`~/data/grid/qgis/styles`. It snapshots the GeoPackage and the `.qml` files into
`volumes/qgs-resources/grid/`, builds `grid_model.qgs` with PyQGIS in a one-off
qgis-server container (`build_project.py`: layers, named styles `default` (voltage)
and `By TSO`, map themes "By voltage" and "By TSO", basemaps, WMS/WFS settings,
print layouts), and calls the config generator. QGIS Server reloads the project
when the file changes; reload the browser page.

After editing `build_project.py` or `publish.sh` in the repo, copy them first:

    scp apps/grid_map/qwc/{build_project.py,publish.sh} cairo-lab:data/qwc/bin/

After editing `themesConfig.json` only, it is enough to regenerate the config:

    docker compose -p tranpy-qwc exec -T qwc-api-gateway \
      curl -sf -X POST 'http://qwc-config-service:9090/generate_configs?tenant=default'

## Checks

    curl -s -o /dev/null -w "%{http_code}\n" http://cairo-lab:8088/
    curl -s "http://cairo-lab:8088/ows/grid_model?SERVICE=WMS&REQUEST=GetCapabilities" | head
    curl -s -o /tmp/m.png "http://cairo-lab:8088/ows/grid_model?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=lines,substations&STYLES=&CRS=EPSG:3857&BBOX=600000,5950000,1700000,7400000&WIDTH=800&HEIGHT=1054&FORMAT=image/png"

## Basemaps and attribution

- Default: Esri World Light Gray Base (`server.arcgisonline.com`), attribution
  "Esri, HERE, Garmin, © OpenStreetMap contributors". Esri's terms apply.
- basemap.de grey (BKG, CC BY 4.0), Germany only.
- OpenStreetMap standard tiles (© OpenStreetMap contributors; OSM tile usage policy).

CARTO Positron, which the QGIS styles name as basemap, now returns
"API KEY REQUIRED" tiles without a key, so it is not used here.

## Notes

- QGIS layer filters on GeoPackages are passed to SQLite: use `LIKE` (case-insensitive
  for ASCII) in search expressions, not `ILIKE`.
- The voltage style is named `default` because QWC2 starts with the style of that
  name (otherwise the alphabetically first); the map theme is still "By voltage".
