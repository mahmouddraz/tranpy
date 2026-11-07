"""Build the QWC2 QGIS project for the German grid model (run with PyQGIS).

Usage (inside the qwc-qgis-server container, see publish.sh):
    python3 build_project.py <gpkg> <styles_dir> <out.qgs>

The GeoPackage must already sit next to (or below) the output project, so the
saved project uses relative data source paths. Every layer gets two named
styles ("By voltage", "By TSO") from ``<layer>_<mode>.qml``; the map themes of
the same names switch between them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qgis.core import (  # noqa: E402
    Qgis,
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsLayerTreeModel,
    QgsLayoutItemLabel,
    QgsLayoutItemLegend,
    QgsLegendStyle,
    QgsLayoutItemMap,
    QgsLayoutItemPicture,
    QgsLayoutItemScaleBar,
    QgsLayoutMeasurement,
    QgsLayoutPoint,
    QgsLayoutSize,
    QgsMapLayerStyle,
    QgsPrintLayout,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsUnitTypes,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import Qt  # noqa: E402
from qgis.PyQt.QtGui import QFont  # noqa: E402

TITLE = "German transmission grid (model v4final)"
# Basemaps (XYZ). Published in WMS only so QWC2 can print them (printLayer);
# they are hidden in the QWC2 layer tree. CARTO Positron now returns
# "API KEY REQUIRED" tiles without a key, hence Esri Light Gray as the default.
BASEMAPS = [
    (
        "esri_light_gray",
        "Grey basemap (Esri World Light Gray)",
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
        "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "16",
        "Esri, HERE, Garmin, © OpenStreetMap contributors",
        "https://www.esri.com/en-us/legal/terms/full-master-agreement",
    ),
    (
        "basemap_de_grey",
        "basemap.de grey (BKG)",
        "https://sgx.geodatenzentrum.de/wmts_basemapde/tile/1.0.0/"
        "de_basemapde_web_raster_grau/default/GLOBAL_WEBMERCATOR/{z}/{y}/{x}.png",
        "18",
        "© basemap.de / BKG (CC BY 4.0)",
        "https://basemap.de",
    ),
    (
        "osm",
        "OpenStreetMap",
        "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "19",
        "© OpenStreetMap contributors",
        "https://www.openstreetmap.org/copyright",
    ),
]
ATTRIBUTION = (
    "Grid model v4final (tranpy). Plants: MaStR (Bundesnetzagentur). "
    "Basemap: Esri, HERE, Garmin, © OpenStreetMap contributors."
)
NORTH_ARROW = "/usr/share/qgis/svg/arrows/NorthArrow_02.svg"
MODES = {"voltage": "By voltage", "tso": "By TSO"}
# Layer style names. QWC2 starts with the style called "default" (else the first one
# alphabetically), so the voltage style is the layer default; the map themes keep
# the readable names "By voltage" / "By TSO".
STYLE_NAMES = {"voltage": "default", "tso": "By TSO"}
# Layer tree, top first (drawing order is bottom-up: bidding zones lowest).
LAYERS = [
    ("neighbours", "Neighbouring networks", True),
    ("plants", "Power plants (MaStR)", True),
    ("plant_aggregates", "Small plants (summed per bus and fuel)", True),
    ("transformers", "Transformers", True),
    ("substations", "Substations", True),
    ("buses", "Buses (model / ANDES ids)", False),
    ("plant_connections", "Plant connections", True),
    ("lines", "Lines (AC circuits, HVDC)", True),
    ("tower_lines", "Tower lines (OpenStreetMap)", True),
    ("control_areas", "TSO control areas (approximate)", True),
    ("bidding_zones", "Bidding zones", True),
]
# Germany in EPSG:3857.
EXTENT = QgsRectangle(600_000, 5_950_000, 1_700_000, 7_400_000)


def gpkg_tables(gpkg: str) -> set[str]:
    import sqlite3

    with sqlite3.connect(gpkg) as con:
        return {r[0] for r in con.execute("select table_name from gpkg_geometry_columns")}


def add_styles(layer: QgsVectorLayer, name: str, styles: Path) -> None:
    """One named style per mode, loaded from ``<layer>_<mode>.qml``."""
    sm = layer.styleManager()
    for i, (mode, style_name) in enumerate(STYLE_NAMES.items()):
        if i == 0:
            sm.renameStyle(sm.currentStyle(), style_name)
        else:
            sm.addStyleFromLayer(style_name)
            sm.setCurrentStyle(style_name)
        qml = styles / f"{name}_{mode}.qml"
        if qml.exists():
            msg, ok = layer.loadNamedStyle(str(qml))
            if not ok:
                print(f"WARN {qml}: {msg}")
        else:
            print(f"WARN missing style {qml}")
    sm.setCurrentStyle(STYLE_NAMES["voltage"])
    assert sorted(sm.styles()) == sorted(STYLE_NAMES.values()), sm.styles()


def add_layout(project: QgsProject, name: str, size: str) -> None:
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(name)
    page = layout.pageCollection().page(0)
    w, h = {"A4": (297.0, 210.0), "A3": (420.0, 297.0)}[size]
    page.setPageSize(QgsLayoutSize(w, h))
    m = 8.0

    title = QgsLayoutItemLabel(layout)
    title.setId("title")
    title.setText(TITLE)
    title.setFont(QFont("DejaVu Sans", 18 if size == "A4" else 24, QFont.Bold))
    title.attemptMove(QgsLayoutPoint(m, m))
    title.attemptResize(QgsLayoutSize(w - 2 * m, 12))
    layout.addLayoutItem(title)

    legend_w = 62 if size == "A4" else 80
    mp = QgsLayoutItemMap(layout)
    mp.setId("map0")
    mp.attemptMove(QgsLayoutPoint(m, m + 14))
    mp.attemptResize(QgsLayoutSize(w - 3 * m - legend_w, h - 2 * m - 22))
    mp.setFrameEnabled(True)
    mp.zoomToExtent(EXTENT)
    layout.addLayoutItem(mp)

    legend = QgsLayoutItemLegend(layout)
    legend.setId("legend")
    legend.setTitle("Legend")
    legend.setLinkedMap(mp)
    legend.setLegendFilterByMapEnabled(True)
    legend.attemptMove(QgsLayoutPoint(w - m - legend_w, m + 14))
    legend.setResizeToContents(False)
    legend.setSymbolHeight(3.0)
    legend.setSymbolWidth(6.0)
    sizes = {
        QgsLegendStyle.Title: 9.0,
        QgsLegendStyle.Group: 7.0,
        QgsLegendStyle.Subgroup: 7.0,
        QgsLegendStyle.SymbolLabel: 6.0,
    }
    for part, pt in sizes.items():
        style = legend.rstyle(part)
        fmt = style.textFormat()
        fmt.setSize(pt if size == "A4" else pt + 1.5)
        style.setTextFormat(fmt)
        style.setMargin(QgsLegendStyle.Top, 0.8)
    legend.attemptResize(QgsLayoutSize(legend_w, h - 2 * m - 40))
    layout.addLayoutItem(legend)

    sb = QgsLayoutItemScaleBar(layout)
    sb.setId("scalebar")
    sb.setStyle("Single Box")
    sb.setLinkedMap(mp)
    sb.applyDefaultSize(QgsUnitTypes.DistanceKilometers)
    sb.setUnitLabel("km")
    sb.attemptMove(QgsLayoutPoint(m + 4, h - m - 18))
    layout.addLayoutItem(sb)

    north = QgsLayoutItemPicture(layout)
    north.setId("north")
    north.setPicturePath(NORTH_ARROW)
    north.attemptMove(QgsLayoutPoint(w - 3 * m - legend_w - 14, m + 18))
    north.attemptResize(QgsLayoutSize(14, 14))
    layout.addLayoutItem(north)

    attr = QgsLayoutItemLabel(layout)
    attr.setId("attribution")
    attr.setText(ATTRIBUTION + " Printed [% format_date(now(), 'yyyy-MM-dd') %].")
    attr.setFont(QFont("DejaVu Sans", 7))
    attr.setHAlign(Qt.AlignRight)
    attr.attemptMove(QgsLayoutPoint(m, h - m - 5))
    attr.attemptResize(QgsLayoutSize(w - 2 * m, 5))
    layout.addLayoutItem(attr)

    project.layoutManager().addLayout(layout)


def main(gpkg: str, styles_dir: str, out: str) -> None:
    app = QgsApplication([], False)
    app.initQgis()
    project = QgsProject.instance()
    project.setTitle(TITLE)
    project.setCrs(QgsCoordinateReferenceSystem("EPSG:3857"))
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    root = project.layerTreeRoot()
    styles = Path(styles_dir)
    present = gpkg_tables(gpkg)

    vector_ids = []
    for name, title, visible in LAYERS:
        if name not in present:
            print(f"skip {name}: not in GeoPackage")
            continue
        layer = QgsVectorLayer(f"{gpkg}|layername={name}", title, "ogr")
        if not layer.isValid():
            raise SystemExit(f"invalid layer {name}")
        layer.setShortName(name)
        add_styles(layer, name, styles)
        project.addMapLayer(layer, False)
        node = root.addLayer(layer)
        node.setItemVisibilityChecked(visible)
        vector_ids.append(layer.id())

    for short, title, url, zmax, attribution, attribution_url in BASEMAPS:
        enc = url.replace("{", "%7B").replace("}", "%7D")
        base = QgsRasterLayer(f"type=xyz&url={enc}&zmax={zmax}&zmin=0", title, "wms")
        if not base.isValid():
            print(f"WARN basemap {short} invalid")
        base.setShortName(short)
        base.setAttribution(attribution)
        base.setAttributionUrl(attribution_url)
        project.addMapLayer(base, False)
        root.addLayer(base).setItemVisibilityChecked(False)

    # Map themes: all named styles switch together.
    model = QgsLayerTreeModel(root)
    themes = project.mapThemeCollection()
    for mode, theme_name in MODES.items():
        for lid in vector_ids:
            project.mapLayer(lid).styleManager().setCurrentStyle(STYLE_NAMES[mode])
        themes.insert(theme_name, themes.createThemeFromCurrentState(root, model))
    for lid in vector_ids:
        project.mapLayer(lid).styleManager().setCurrentStyle(STYLE_NAMES["voltage"])

    # QGIS Server / OWS.
    project.writeEntryBool("WMSServiceCapabilities", "/", True)
    project.writeEntry("WMSServiceTitle", "/", TITLE)
    project.writeEntry("WMSServiceAbstract", "/", ATTRIBUTION)
    project.writeEntry("WMSContactOrganization", "/", "tranpy")
    project.writeEntry("WMSKeywordList", "/", ["transmission grid", "Germany"])
    project.writeEntry("WMSCrsList", "/", ["EPSG:3857", "EPSG:4326", "EPSG:25832"])
    project.writeEntry(
        "WMSExtent",
        "/",
        [str(v) for v in (EXTENT.xMinimum(), EXTENT.yMinimum(), EXTENT.xMaximum(), EXTENT.yMaximum())],
    )
    project.writeEntryBool("WMSAddWktGeometry", "/", True)
    project.writeEntryBool("WMSFeatureInfoUseAttributeFormSettings", "/", False)
    project.writeEntry("WMSPrecision", "/", "6")
    project.writeEntry("WMSMaxAtlasFeatures", "/", 1)
    project.writeEntry("WFSLayers", "/", vector_ids)
    for lid in vector_ids:
        project.writeEntry("WFSLayersPrecision", "/" + lid, 6)
    project.writeEntryBool("WMSUseLayerIDs", "/", False)
    project.viewSettings().setDefaultViewExtent(
        __import__("qgis.core", fromlist=["QgsReferencedRectangle"]).QgsReferencedRectangle(
            EXTENT, project.crs()
        )
    )

    add_layout(project, "A4 Landscape", "A4")
    add_layout(project, "A3 Landscape", "A3")

    if not project.write(out):
        raise SystemExit(f"could not write {out}")
    print(f"wrote {out}: {len(vector_ids)} vector layers, themes {themes.mapThemes()}")
    app.exitQgis()


if __name__ == "__main__":
    main(*sys.argv[1:4])
