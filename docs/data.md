# Data management

How TranPy stores, publishes and uses its datasets, and why. The guiding rules: raw data
are kept bit-exact, every file is pinned by a SHA-256 checksum, nothing large lives in the
repository or has to live on a user's disk, and analyses stream the data.

## Datasets

| Dataset | Content | Size | Where |
|---|---|---|---|
| Snapshots, 9-bus and 39-bus | 5,000 events each, one feature vector per event | 3 MB | bundled with the package |
| Events, 500 per system | full trajectories of all recorded variables | about 100 MB per system (compact) | converted from the recorded CSVs |
| Events, full sets | 10,000 (9-bus) and 8,000 (39-bus) events | about 2 GB per system (compact) | sharded; GitHub release assets |
| Pretrained pipelines | 18 models x 2 systems, skops format | 277 MB | GitHub release assets |

The original PowerFactory CSV exports total about 19 GB. They are the raw archive and are
not needed for analysis once converted.

## Storage format

* **One sample format.** `SignalDataset` holds all events back to back with offsets; on disk
  it is a `.npz` file with a JSON header, read with `allow_pickle=False` (loading data can
  never execute code).
* **Lossless compact encoding.** PowerFactory exports carry at most 6 decimals. Such values
  are stored as scaled integers, differenced along time and byte-shuffled before
  compression. Decoding divides exact integers by an exact power of ten, which is correctly
  rounded, so the result is **bit-for-bit identical** to parsing the CSV. Files are 2.6 to
  2.9 times smaller than plain float64. Data that are not short decimals (e.g. simulator
  output in memory) fall back to float64 automatically.
* **Why not float32.** Halving the precision changes derived features by up to 3 % (rotor
  speed stays near 1.0 p.u. with deviations around 1e-5, below float32 resolution). Labels
  and fault timing would survive, but the features would not; the archive must be exact.
* **Reproducible files.** Archive members are written in a fixed order with fixed
  timestamps, so re-running a conversion reproduces the file and its checksum.

## Shards and manifests

Large datasets are split into shards of 250 events (about 50 MB each) and described by a
`manifest.json`:

```text
{"manifest_version": 1, "name": "ieee_9bus_events",
 "channel_names": [...], "label_names": ["stable", "unstable"],
 "base_url": null,
 "metadata": {"system": "ieee_9bus", "label_counts": {...}},
 "shards": [{"filename": "ieee_9bus_events-00000.npz", "sha256": "...",
             "n_samples": 250, "sample_ids": [0, 249], "url": "..."}]}
```

* `iter_shards` yields one verified shard at a time. Shards in the manifest's local folder
  are read in place; others are downloaded to the user cache, verified, used and deleted,
  so local disk never holds more than one shard.
* `extract_window_tables` makes **one pass** over the shards and builds the feature and
  sequence tables for every observation window at once. The tables are small (10,000
  events x 165 features is 13 MB) and can be cached on disk, keyed by the shard checksums
  and every extraction setting, so later model runs never touch the raw data again.
* `ShardWriter` writes shards resumably: interrupted conversions continue where they
  stopped.

## Hosting

Code lives in the GitHub repository; data and pretrained models are **assets of GitHub
releases** of the same repository (tag `data-v1`, `data-v2`, ...), the same approach as
many scientific Python packages (SciPy, scikit-image and others fetch data the same way).
Assets may be up to 2 GB each (shards are about 50 MB) with no practical limit on total
size or downloads, and their URLs are predictable:
`https://github.com/mahmouddraz/tranpy/releases/download/<tag>/<file>`.

Files are never committed to git (the repository stays small) and Google Drive is no
longer used (quota limits, HTML interstitials, no stable direct links).

Publishing a release:

```bash
python scripts/publish_github_release.py prepare data-v1 \
    --events ieee_9bus=shards/ieee_9bus new_england=shards/new_england \
    --pretrained pretrained/
gh release create data-v1 --repo mahmouddraz/tranpy --title "Data data-v1" --notes-file release/NOTES.md
gh release upload data-v1 --repo mahmouddraz/tranpy release/*.npz release/*.json release/*.skops
```

`prepare` writes the final URLs into the manifests and the URLs and SHA-256 checksums into
`src/tranpy/resources/registry.json`; commit that file with the release. Afterwards
`tranpy.open_event_shards("ieee_9bus")` and `load_pretrained(...)` work for anyone without
local data. A new simulation campaign gets a new tag; old tags stay, so earlier results
remain reproducible.

## Workflows

```bash
# PowerFactory CSV folder -> shards (bounded memory, resumable)
tranpy build-shards CSV_DIR new_england shards/new_england --shard-size 250

# legacy Google Drive files -> shards (downloads a batch, converts, deletes)
python scripts/migrate_drive_events.py ieee_9bus INDEX_JSON shards/ieee_9bus
```

```python
import tranpy
from tranpy.early_prediction import extract_window_tables, evaluate_window_tables

events = tranpy.open_event_shards("ieee_9bus", source="shards/ieee_9bus")  # no source: release
tables = extract_window_tables(events, reference="fault_start", windows_s=(0.1, 0.2, 0.4),
                               sequences=True, cache_directory="cache/tables")
results = evaluate_window_tables(tables, feature_models, sequence_models=networks, n_repeats=3)
```

## Provenance

* Snapshot datasets: converted from the 0.1 pickles by `scripts/convert_legacy_data.py`;
  source file checksums are stored in each dataset's metadata.
* Event datasets: PowerFactory RMS simulations exported as CSV (two header rows: element,
  variable). Labels: any generator out of step at any time. Fault start and clearing are
  detected from bus voltages (start is causal; clearing is not, see `validation.md`, D9).
* The Google Drive copies are legacy sources only; the package no longer downloads from
  Google Drive.
