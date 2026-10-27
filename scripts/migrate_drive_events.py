"""Stream the legacy Google Drive event CSVs into a sharded dataset (one-time migration).

    python scripts/migrate_drive_events.py SYSTEM INDEX_JSON OUT_DIR [--shard-size 250]
        [--workers 8] [--limit N]

INDEX_JSON maps ``results_event_<id>.csv`` to public Google Drive file ids (the
``_file_index.json`` written by tranpy 0.1). For each batch of ``shard-size`` events the
CSVs are downloaded in parallel into a temporary folder, converted into one shard and
deleted, so local disk holds at most one batch of CSVs plus the compact shards. The run
is resumable: events already in OUT_DIR are skipped. Files that fail after retries are
listed in OUT_DIR/failed.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tranpy.datasets import read_event_csvs
from tranpy.systems import get_system
from tranpy_core.data import ShardWriter

logger = logging.getLogger("migrate")
_NAME = re.compile(r"^results_event_(\d+)\.csv$")
_URL = "https://drive.google.com/uc?export=download&id={}"


def download(file_id: str, target: Path, retries: int = 4) -> bool:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(_URL.format(file_id), timeout=120) as response:
                data = response.read()
            if data[:20].lstrip().lower().startswith((b"<!doctype", b"<html")):
                raise OSError("Google Drive returned an HTML page instead of the file")
            target.write_bytes(data)
            return True
        except OSError as exc:
            logger.warning("%s attempt %d failed: %s", target.name, attempt + 1, exc)
            time.sleep(2**attempt)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("system")
    parser.add_argument("index", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--shard-size", type=int, default=250)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="stop after this many events")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    spec = get_system(args.system)
    index = {
        int(m.group(1)): fid
        for name, fid in json.loads(args.index.read_text()).items()
        if (m := _NAME.match(name))
    }
    writer = ShardWriter(
        args.out_dir,
        name=f"{spec.key}_events",
        shard_size=args.shard_size,
        metadata={
            "system": spec.key,
            "source": "PowerFactory RMS simulation (CSV export, Google Drive)",
        },
    )
    done = writer.done_sample_ids()
    todo = [i for i in sorted(index) if i not in done][: args.limit]
    failed_path = args.out_dir / "failed.json"
    failed: list[int] = json.loads(failed_path.read_text()) if failed_path.exists() else []
    logger.info("%s: %d indexed, %d done, %d to do", spec.key, len(index), len(done), len(todo))
    for start in range(0, len(todo), args.shard_size):
        batch = todo[start : start + args.shard_size]
        with tempfile.TemporaryDirectory(prefix="tranpy-drive-") as tmp:
            tmp_dir = Path(tmp)
            with ThreadPoolExecutor(args.workers) as pool:
                targets = [tmp_dir / f"results_event_{i}.csv" for i in batch]
                ok = list(pool.map(download, [index[i] for i in batch], targets))
            got = [i for i, success in zip(batch, ok, strict=True) if success]
            failed.extend(i for i, success in zip(batch, ok, strict=True) if not success)
            if got:
                writer.add(read_event_csvs(tmp_dir, spec, event_ids=got))
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.write_text(json.dumps(sorted(set(failed))))
        logger.info("%d / %d events processed", min(start + args.shard_size, len(todo)), len(todo))
    manifest = writer.close()
    logger.info(
        "done: %d shards, %d events, %d failed",
        len(manifest.shards),
        manifest.n_samples,
        len(set(failed)),
    )


if __name__ == "__main__":
    main()
