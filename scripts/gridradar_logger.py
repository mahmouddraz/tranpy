"""Log the free Gridradar CE median frequency (1 s) and network time without gaps.

The free account returns at most the last hour and no time range; each request continues
after the last value the account received. Run this every 30 minutes (cron) and it keeps a
gapless series: one parquet file per metric and UTC day under ``--out``.

The token is read from ``GRIDRADAR_API_TOKEN`` or from the env file given by ``--env``
(``KEY=value`` lines); it is never printed. Gridradar data is for research use only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://api.gridradar.net/query"
METRICS = ("frequency-ucte-median-1s", "net-time")


def _token(env_file: Path | None) -> str:
    token = os.environ.get("GRIDRADAR_API_TOKEN", "")
    if not token and env_file and env_file.exists():
        for line in env_file.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "GRIDRADAR_API_TOKEN":
                token = value.strip().strip("'\"")
    if not token:
        sys.exit("GRIDRADAR_API_TOKEN not set (env or --env file)")
    return token


def fetch(metric: str, token: str) -> pd.DataFrame:
    body = json.dumps({"metric": metric, "format": "json", "ts": "rfc3339", "aggr": "1s"})
    req = urllib.request.Request(
        URL,
        data=body.encode(),
        headers={"Content-type": "application/json", "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read())
    if isinstance(data, dict):  # {"columns": ["code","msg"], "data": [[code, msg]]}
        raise RuntimeError(f"{metric}: {data.get('data')}")
    points = data[0]["datapoints"] if data else []
    df = pd.DataFrame(points, columns=["value", "time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df[["time", "value"]]


def store(df: pd.DataFrame, out: Path, metric: str) -> int:
    new = 0
    for day, part in df.groupby(df["time"].dt.strftime("%Y-%m-%d")):
        path = out / metric / f"{day}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        old = pd.read_parquet(path) if path.exists() else part.iloc[:0]
        merged = pd.concat([old, part]).drop_duplicates("time").sort_values("time")
        new += len(merged) - len(old)
        merged.to_parquet(path, index=False)
    return new


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path.home() / "data" / "gridradar")
    p.add_argument("--env", type=Path, default=Path.home() / ".config" / "tranpy" / "gridradar.env")
    a = p.parse_args()
    token = _token(a.env)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for i, metric in enumerate(METRICS):
        if i:
            time.sleep(11)  # minimum request interval is 10 s per metric
        try:
            df = fetch(metric, token)
            n = store(df, a.out, metric)
            last = df["time"].max() if len(df) else None
            print(f"{stamp} {metric}: {len(df)} received, {n} new, last {last}", flush=True)
        except Exception as exc:  # keep the other metric and the next cron run alive
            print(f"{stamp} {metric}: ERROR {exc}", flush=True)


if __name__ == "__main__":
    main()
