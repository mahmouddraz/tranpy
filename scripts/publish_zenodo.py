"""Publish a sharded event dataset on Zenodo and register it in the package.

    ZENODO_TOKEN=... python scripts/publish_zenodo.py FOLDER SYSTEM --title "..." [--sandbox]

Creates a Zenodo *draft* (nothing is public until you press Publish on zenodo.org), writes
the record's download URLs into FOLDER/manifest.json, uploads the shards and the manifest,
and records the manifest URL and SHA-256 under ``events.<SYSTEM>`` in
src/tranpy/resources/registry.json. Zenodo reserves the record id when the draft is
created, so the URLs are final; they start working once the record is published. The
token (scope ``deposit:write``) is read from the environment only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

from tranpy_core.data import ShardManifest, sha256_file

REGISTRY = Path(__file__).resolve().parents[1] / "src" / "tranpy" / "resources" / "registry.json"


def _api(sandbox: bool) -> str:
    return "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"


def _request(
    method: str, url: str, token: str, body: bytes | None = None, json_body: Any = None
) -> Any:
    headers = {"Authorization": f"Bearer {token}"}
    if json_body is not None:
        body = json.dumps(json_body).encode()
        headers["Content-Type"] = "application/json"
    elif body is not None:
        headers["Content-Type"] = "application/octet-stream"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as response:
        text = response.read()
    return json.loads(text) if text else None


def publish(folder: Path, system: str, title: str, sandbox: bool) -> None:
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit("set ZENODO_TOKEN (scope deposit:write)")
    manifest = ShardManifest.load(folder / "manifest.json")
    api = _api(sandbox)
    deposition = _request("POST", f"{api}/deposit/depositions", token, json_body={})
    record_id = deposition["id"]
    host = "https://sandbox.zenodo.org" if sandbox else "https://zenodo.org"
    base_url = f"{host}/records/{record_id}/files"
    data = manifest.to_dict()
    data["base_url"] = base_url
    data["shards"] = [
        {**s, "url": f"{base_url}/{s['filename']}?download=1"} for s in data["shards"]
    ]
    (folder / "manifest.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    bucket = deposition["links"]["bucket"]
    names = [s.filename for s in manifest.shards] + ["manifest.json"]
    for name in names:
        path = folder / name
        print(f"uploading {name} ({path.stat().st_size / 1e6:.1f} MB)")
        _request("PUT", f"{bucket}/{name}", token, body=path.read_bytes())
    metadata = {
        "metadata": {
            "title": title,
            "upload_type": "dataset",
            "description": "Simulated fault-event trajectories for transient stability analysis "
            f"({manifest.n_samples} events; TranPy sharded format: manifest.json plus .npz shards "
            "with SHA-256 checksums).",
            "creators": [{"name": "Draz, Mahmoud"}],
            "license": "cc-by-4.0",
        }
    }
    _request("PUT", deposition["links"]["self"], token, json_body=metadata)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    registry["events"][system].update(
        manifest_url=f"{base_url}/manifest.json?download=1",
        manifest_sha256=sha256_file(folder / "manifest.json"),
        n_samples=manifest.n_samples,
        zenodo_record=str(record_id),
    )
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(f"draft {record_id}: {deposition['links']['html']}")
    print(f"registry updated for {system}; review the draft and press Publish on Zenodo")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", type=Path)
    parser.add_argument("system")
    parser.add_argument("--title", required=True)
    parser.add_argument("--sandbox", action="store_true")
    args = parser.parse_args()
    publish(args.folder, args.system, args.title, args.sandbox)


if __name__ == "__main__":
    sys.exit(main())
