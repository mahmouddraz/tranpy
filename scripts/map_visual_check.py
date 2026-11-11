"""Visual check of every station and junction of a served grid map.

Renders each substation/tee/junction of an export at a fixed zoom from a served map
(MapLibre in headless Chromium, several pages in parallel), counts the detector findings of
``scripts/map_acceptance.py`` (``events.csv``) within ``--radius-m`` of each place, and writes:

- ``tiles/<id>.png``: one tile per place (centre crosshair, side panel hidden);
- ``places.csv``: one row per place with the finding counts per check and a score;
- ``flags/<id>.csv``: the findings within the radius of each place;
- ``ranked.csv``: places worst first;
- ``sheets/sheet_NNN.png``: contact sheets (``--grid`` x ``--grid``, worst first, captioned);
- ``index.html``: all tiles with name and flags, worst first.

Usage (Playwright with Chromium and Pillow needed)::

    python scripts/map_visual_check.py --url http://localhost:8094/ \
        --export ~/data/grid/staging_osmgraph/web \
        --events ~/data/grid/acceptance/staging_v1_geom/events.csv \
        --out ~/data/grid/acceptance/visual_staging_v1
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

CHECKS = ("on_own_way", "no_zigzag", "no_loop", "no_spike", "station_end")
WEIGHT = {"on_own_way": 1.0, "no_zigzag": 2.0, "no_loop": 3.0, "no_spike": 2.0, "station_end": 2.0}

HIDE_CSS = """
#panel, aside, .maplibregl-ctrl-top-right { display: none !important; }
body::after { content: ""; position: fixed; left: 50%; top: 50%; width: 18px; height: 18px;
  margin: -9px 0 0 -9px; border: 2px solid #1565c0; border-radius: 50%; pointer-events: none;
  z-index: 9999; }
"""

WAIT_JS = """([lon, lat, z]) => new Promise((res) => {
  const m = window.gridMap; m.jumpTo({center: [lon, lat], zoom: z});
  const t0 = Date.now();
  const f = () => { if ((m.loaded() && m.areTilesLoaded()) || Date.now() - t0 > 8000) res();
                    else setTimeout(f, 150); };
  setTimeout(f, 300);
})"""


def read_places(export: str | Path) -> pd.DataFrame:
    """Substations, tees and junctions of an export (web dir or GeoPackage)."""
    p = Path(export).expanduser()
    sub = (
        gpd.read_file(p / "substations.geojson")
        if p.is_dir()
        else gpd.read_file(p, layer="substations")
    ).to_crs(4326)
    out = pd.DataFrame(
        {
            "site": sub["site"].astype(str),
            "name": sub.get("name", sub["site"]).astype(str),
            "kind": sub.get("kind", "substation").astype(str),
            "lon": sub.geometry.x.round(6),
            "lat": sub.geometry.y.round(6),
        }
    )
    out = out.sort_values(["kind", "site"]).reset_index(drop=True)
    out.insert(0, "id", [f"S{i + 1:04d}" for i in range(len(out))])
    return out


def flag_places(
    places: pd.DataFrame, events: pd.DataFrame, radius_m: float
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Counts per check of the findings within ``radius_m`` of each place, and the findings."""
    near: dict[str, pd.DataFrame] = {}
    rows = []
    ev_lon = events["lon"].to_numpy(float) if len(events) else np.zeros(0)
    ev_lat = events["lat"].to_numpy(float) if len(events) else np.zeros(0)
    for r in places.itertuples():
        kx = 111_320 * math.cos(math.radians(r.lat))
        d = np.hypot((ev_lon - r.lon) * kx, (ev_lat - r.lat) * 110_574)
        hit = events[d <= radius_m].assign(distance_m=d[d <= radius_m].round())
        near[r.id] = hit
        counts = hit["check"].value_counts() if len(hit) else pd.Series(dtype=int)
        row = {c: int(counts.get(c, 0)) for c in CHECKS}
        row["circuits_flagged"] = int(hit["mrid"].nunique()) if len(hit) else 0
        row["score"] = round(sum(WEIGHT[c] * row[c] for c in CHECKS), 1)
        rows.append(row)
    return pd.concat([places, pd.DataFrame(rows)], axis=1), near


async def _shoot(url: str, places: pd.DataFrame, out: Path, zoom: float, size: int, pages: int):
    from playwright.async_api import async_playwright

    out.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue = asyncio.Queue()
    for r in places.itertuples():
        if not (out / f"{r.id}.png").exists():
            queue.put_nowait(r)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"]
        )

        async def worker() -> None:
            page = await browser.new_page(viewport={"width": size, "height": size})
            await page.goto(url)
            await page.wait_for_selector("body[data-ready='1']", timeout=180_000)
            await page.add_style_tag(content=HIDE_CSS)
            while not queue.empty():
                r = queue.get_nowait()
                await page.evaluate(WAIT_JS, [float(r.lon), float(r.lat), zoom])
                await page.screenshot(path=str(out / f"{r.id}.png"))
            await page.close()

        await asyncio.gather(*(worker() for _ in range(pages)))
        await browser.close()


def _caption(r: pd.Series) -> str:
    flags = ", ".join(f"{c.replace('no_', '')} {r[c]}" for c in CHECKS if r[c])
    name = str(r["name"]).split(";")[0][:40]
    return f"{r['id']} {name} ({r['kind']}) score {r['score']}" + (
        f"\n{flags}" if flags else "\nno flags"
    )


def contact_sheets(ranked: pd.DataFrame, tiles: Path, out: Path, grid: int, thumb: int) -> int:
    from PIL import Image, ImageDraw

    out.mkdir(parents=True, exist_ok=True)
    cap_h = 34
    per = grid * grid
    n = 0
    for start in range(0, len(ranked), per):
        sheet = Image.new("RGB", (grid * thumb, grid * (thumb + cap_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, (_, r) in enumerate(ranked.iloc[start : start + per].iterrows()):
            x, y = (k % grid) * thumb, (k // grid) * (thumb + cap_h)
            f = tiles / f"{r['id']}.png"
            if f.exists():
                sheet.paste(Image.open(f).convert("RGB").resize((thumb, thumb)), (x, y + cap_h))
            colour = "#b71c1c" if r["score"] > 0 else "#1b5e20"
            draw.text((x + 4, y + 2), _caption(r), fill=colour)
            draw.rectangle([x, y, x + thumb - 1, y + thumb + cap_h - 1], outline="#9e9e9e")
        n += 1
        sheet.save(out / f"sheet_{n:03d}.png")
    return n


def index_html(ranked: pd.DataFrame, url: str, zoom: float) -> str:
    cards = []
    for _, r in ranked.iterrows():
        flags = " · ".join(f"{c} {r[c]}" for c in CHECKS if r[c]) or "no flags"
        link = f"{url}#{zoom}/{r['lat']}/{r['lon']}"
        cards.append(
            f'<figure class="{"bad" if r["score"] > 0 else "ok"}"><a href="tiles/{r["id"]}.png">'
            f'<img loading="lazy" src="tiles/{r["id"]}.png" alt=""></a><figcaption>'
            f"<b>{html.escape(str(r['id']))} {html.escape(str(r['name']))}</b> "
            f"({html.escape(str(r['kind']))}) score {r['score']}<br>{html.escape(flags)}<br>"
            f'<a href="flags/{r["id"]}.csv">flags</a> · {r["lon"]}, {r["lat"]} · '
            f'<a href="{html.escape(link)}">map</a></figcaption></figure>'
        )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Station visual check</title><style>
body{{font:13px system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}}
figure{{margin:0;background:#fff;border:1px solid #ddd}} figure.bad{{border-color:#c62828}}
img{{width:100%;display:block}} figcaption{{padding:4px 6px}}</style></head><body>
<h1>Station visual check</h1><p>{len(ranked)} places, worst first; map {html.escape(url)},
zoom {zoom}; flags = detector findings within the radius (scripts/map_acceptance.py).</p>
<main>{"".join(cards)}</main></body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--export", required=True, help="web dir or grid_model.gpkg of the map")
    ap.add_argument("--events", required=True, help="events.csv of map_acceptance.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--zoom", type=float, default=15.0)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--pages", type=int, default=4)
    ap.add_argument("--radius-m", type=float, default=1500.0)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--thumb", type=int, default=400)
    ap.add_argument("--no-shots", action="store_true", help="reuse existing tiles")
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    places = read_places(a.export)
    events = pd.read_csv(Path(a.events).expanduser())
    flagged, near = flag_places(places, events, a.radius_m)
    flagged.to_csv(out / "places.csv", index=False)
    (out / "flags").mkdir(exist_ok=True)
    for pid, hit in near.items():
        hit.to_csv(out / "flags" / f"{pid}.csv", index=False)
    ranked = flagged.sort_values(["score", "circuits_flagged"], ascending=False)
    ranked.to_csv(out / "ranked.csv", index=False)
    if not a.no_shots:
        asyncio.run(_shoot(a.url, places, out / "tiles", a.zoom, a.size, a.pages))
    n = contact_sheets(ranked, out / "tiles", out / "sheets", a.grid, a.thumb)
    (out / "index.html").write_text(index_html(ranked, a.url, a.zoom))
    summary = {
        "url": a.url,
        "places": len(places),
        "flagged": int((flagged["score"] > 0).sum()),
        "sheets": n,
        "by_kind": places["kind"].value_counts().to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
