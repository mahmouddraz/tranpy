"""Load a built QGIS project with PyQGIS and report whether it and every layer are valid.

Usage (inside the qwc-qgis-server container, see scripts/publish_map.sh):
    python3 check_project.py <project.qgs> <report.json>
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qgis.core import QgsApplication, QgsProject, QgsVectorLayer


def main(qgs: str, report: str) -> int:
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    ok = project.read(qgs)
    layers = list(project.mapLayers().values())
    vec = [lyr for lyr in layers if isinstance(lyr, QgsVectorLayer)]
    res = {
        "project": qgs,
        "valid": bool(ok),
        "layers": len(vec),
        "invalid_layers": [lyr.name() for lyr in vec if not lyr.isValid()],
        "features": {lyr.name(): lyr.featureCount() for lyr in vec},
        "themes": project.mapThemeCollection().mapThemes(),
    }
    with open(report, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("valid", "layers", "invalid_layers", "themes")}))
    app.exitQgis()
    return 0 if ok and not res["invalid_layers"] else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
