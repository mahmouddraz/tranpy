#!/usr/bin/env bash
# Serve the live web map (8080) from the published folder ~/data/grid/live_map, not from the
# code checkout. Run once on cairo-lab (the owner's step; it restarts the 8080 server):
#     bash ~/code/tranpy/scripts/serve_live.sh
# live_map/ holds the published app files (app.js, index.html, style.css, style.js,
# symbols/) and data -> ~/data/grid/web (the published export). scripts/publish_map.sh
# copies staging into these two places; editing the checkout no longer changes 8080.
set -euo pipefail
LIVE=${LIVE:-$HOME/data/grid/live_map}
PORT=${PORT:-8080}
test -f "$LIVE/app.js" && test -e "$LIVE/data/metadata.json"
old=$(ss -ltnp | grep ":$PORT " | grep -o 'pid=[0-9]*' | cut -d= -f2 || true)
[ -n "$old" ] && kill "$old" && sleep 1
cd "$LIVE"
nohup python3 -m http.server "$PORT" --bind 0.0.0.0 > "$HOME/data/grid/live_map.http.log" 2>&1 < /dev/null &
sleep 2
curl -sf "http://localhost:$PORT/data/metadata.json" | grep -E 'source_file|source_sha256'
echo "8080 now serves $LIVE"
