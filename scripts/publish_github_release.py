"""Publish datasets and pretrained models as assets of a GitHub release.

    python scripts/publish_github_release.py prepare TAG [--repo OWNER/NAME]
        [--events SYSTEM=FOLDER ...] [--pretrained FOLDER]
    gh release create TAG --repo OWNER/NAME --title "Data TAG" --notes-file release/NOTES.md
    gh release upload TAG --repo OWNER/NAME release/*.npz release/*.json release/*.skops

``prepare`` collects every file to publish in ./release/, writes the final download URLs
(``https://github.com/OWNER/NAME/releases/download/TAG/<file>``) into the shard manifests,
and records URLs and SHA-256 checksums in src/tranpy/resources/registry.json. Release
URLs are known in advance, so the registry is final before anything is uploaded; commit
it together with the release. The two ``gh`` commands (run with an account that can
push to the repository) create the release and upload the files.

Asset names must be unique within a release: shards are prefixed with their dataset
name, manifests are renamed ``<dataset>-manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from tranpy.systems import get_system
from tranpy_core.data import ShardManifest, sha256_file

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "src" / "tranpy" / "resources" / "registry.json"


def _link(source: Path, target: Path) -> None:
    """Hard-link (no extra disk space); copy if linking is not possible."""
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def release_url(repo: str, tag: str, filename: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{filename}"


def prepare_events(
    system: str, folder: Path, out: Path, repo: str, tag: str, registry: dict
) -> list[str]:
    key = get_system(system).key
    manifest = ShardManifest.load(folder / "manifest.json")
    data = manifest.to_dict()
    data["base_url"] = None
    data["shards"] = [{**s, "url": release_url(repo, tag, s["filename"])} for s in data["shards"]]
    for shard in manifest.shards:
        _link(folder / shard.filename, out / shard.filename)
    manifest_name = f"{manifest.name}-manifest.json"
    (out / manifest_name).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    registry["events"][key].update(
        manifest_url=release_url(repo, tag, manifest_name),
        manifest_sha256=sha256_file(out / manifest_name),
        n_samples=manifest.n_samples,
        release=tag,
    )
    return [
        f"- `{manifest_name}`: {manifest.n_samples} {key} events in {len(manifest.shards)} shards"
    ]


def prepare_pretrained(folder: Path, out: Path, repo: str, tag: str, registry: dict) -> list[str]:
    entries = json.loads((folder / "registry_entries.json").read_text(encoding="utf-8"))
    for name, entry in entries.items():
        _link(folder / entry["filename"], out / entry["filename"])
        if sha256_file(out / entry["filename"]) != entry["sha256"]:
            raise SystemExit(f"{entry['filename']} does not match its recorded checksum")
        registry["pretrained"][name] = {
            **entry,
            "url": release_url(repo, tag, entry["filename"]),
            "release": tag,
        }
    return [f"- {len(entries)} pretrained pipelines (skops)"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("tag")
    prep.add_argument("--repo", default="mahmouddraz/tranpy")
    prep.add_argument("--events", nargs="*", default=[], metavar="SYSTEM=FOLDER")
    prep.add_argument("--pretrained", type=Path)
    prep.add_argument("--out", type=Path, default=Path("release"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    notes = [f"# Data release {args.tag}", "", "Files are verified by SHA-256 on download.", ""]
    for item in args.events:
        system, folder = item.split("=", 1)
        notes += prepare_events(system, Path(folder), args.out, args.repo, args.tag, registry)
    if args.pretrained:
        notes += prepare_pretrained(args.pretrained, args.out, args.repo, args.tag, registry)
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    (args.out / "NOTES.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    files = sorted(p.name for p in args.out.iterdir() if p.name != "NOTES.md")
    size = sum((args.out / f).stat().st_size for f in files) / 1e9
    print(f"{len(files)} files ({size:.2f} GB) in {args.out}/; registry updated")
    print(
        f"next: gh release create {args.tag} --repo {args.repo} --title 'Data {args.tag}' "
        f"--notes-file {args.out}/NOTES.md && gh release upload {args.tag} --repo {args.repo} "
        f"{args.out}/*.npz {args.out}/*.json {args.out}/*.skops"
    )


if __name__ == "__main__":
    main()
