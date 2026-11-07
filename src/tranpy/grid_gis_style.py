"""Cartography of the grid GIS export: QGIS styles, QGIS project, SVG symbols, web palette.

One specification (:data:`VOLTAGE_COLOURS`, :data:`FUEL_SYMBOLS`, the TSO colours of
:mod:`tranpy.grid_gis`) drives everything, so QGIS and the web viewer look the same:

* :func:`write_qgis_styles` - ``.qml`` per layer and mode (QGIS 3 rule-based renderers,
  scale-dependent rule-based labels);
* :func:`write_qgis_project` - ``grid_model.qgs`` referencing ``grid_model.gpkg`` next to it
  (relative paths) with a neutral XYZ basemap; the "By voltage" and "By TSO" styles are
  stored as named layer styles (Layer > Styles in QGIS);
* :func:`write_svg_symbols` and :func:`web_style` - symbols and colours for the web viewer.

Conventions follow the ENTSO-E grid map: 380/400 kV red, 220 kV green, HVDC dashed blue;
substations as squares (filled for 380 kV, outline for 220 kV), tee/junction points as rings.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from tranpy.grid_gis import FUEL_CATEGORY, PLANT_INDIVIDUAL_MW, TSO_COLOURS

__all__ = [
    "BASEMAP",
    "FUEL_SYMBOLS",
    "VOLTAGE_COLOURS",
    "layer_style",
    "web_style",
    "write_all",
    "write_qgis_project",
    "write_qgis_styles",
    "write_svg_symbols",
]

QGIS_VERSION = "3.34.0-Prizren"

VOLTAGE_COLOURS = {"380": "#d7191c", "220": "#1a9641", "hvdc": "#2166ac", "110": "#555555"}
"""ENTSO-E grid-map convention: 380/400 kV red, 220 kV green, HVDC blue (dashed)."""

OUT_OF_SERVICE = "#9e9e9e"
INTERCONNECTOR = "#6a3d9a"
NEUTRAL = "#455a64"

FUEL_SYMBOLS: dict[str, tuple[str, str, str, str]] = {
    "nuclear": ("Nuclear", "circle", "#8e24aa", "N"),
    "lignite": ("Lignite", "square", "#795548", "L"),
    "hard_coal": ("Hard coal", "square", "#37474f", "C"),
    "gas": ("Gas", "triangle", "#ef6c00", "G"),
    "oil": ("Oil", "triangle", "#827717", "O"),
    "hydro": ("Hydro (run-of-river, reservoir)", "hexagon", "#1e88e5", "H"),
    "pumped_storage": ("Pumped storage", "hexagon", "#0d47a1", "P"),
    "storage": ("Battery and other storage", "diamond", "#00897b", "S"),
    "biomass": ("Biomass", "circle", "#7cb342", "B"),
    "waste": ("Waste", "diamond", "#8d6e63", "W"),
    "other": ("Other", "circle", "#9e9e9e", "X"),
}
"""Fuel category -> (legend label, marker shape, colour, letter)."""

BASEMAP: dict[str, str] = {
    "name": "CARTO Positron (no labels)",
    "url": "https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png",
    "attribution": "© OpenStreetMap contributors, © CARTO",
    "zmax": "19",
}
"""Neutral raster basemap of the QGIS project (CARTO basemap terms apply)."""

LINE_ZOOM: tuple[tuple[float, float, float, float], ...] = (
    (5, 0.9, 0.6, 1.4),
    (8, 1.8, 1.2, 3.2),
    (11, 2.8, 1.9, 5.6),
    (14, 4.0, 2.7, 8.5),
)
"""Web map: (zoom, 380 kV width px, 220 kV width px, spacing px of parallel circuits)."""
PLANT_SIZE_STOPS = ((0, 0.4), (7, 0.55), (20, 0.9), (45, 1.3), (75, 1.65))
"""Web map: sqrt(MW) -> icon scale of a plant (size proportional to sqrt(MW), clamped)."""
PLANT_LABEL_MW = 300
KV_CLASSES = ("380", "220", "110")
"""Voltage classes of the transformer symbols (HV circle, LV circle)."""

TSO_CODES = ("50Hertz", "Amprion", "TenneT", "TransnetBW")
_ZOOM_SCALE = {6: 9_000_000, 7: 4_500_000, 8: 2_200_000, 9: 1_100_000, 10: 550_000, 11: 280_000}
"""Web-map zoom -> approximate scale denominator (at 51 deg N)."""


# --------------------------------------------------------------------------- helpers
def _rgba(hex_colour: str, alpha: float = 1.0) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"{r},{g},{b},{round(alpha * 255)}"


def _options(parent: ET.Element, values: dict[str, Any]) -> ET.Element:
    opt = ET.SubElement(parent, "Option", type="Map")
    for key, value in values.items():
        ET.SubElement(opt, "Option", name=key, type="QString", value=str(value))
    return opt


def _ddp(parent: ET.Element, properties: dict[str, str]) -> None:
    """Data-defined properties (expression per property name) of a symbol layer."""
    dd = ET.SubElement(parent, "data_defined_properties")
    root = ET.SubElement(dd, "Option", type="Map")
    ET.SubElement(root, "Option", name="name", type="QString", value="")
    props = ET.SubElement(root, "Option", name="properties", type="Map")
    for key, expression in properties.items():
        p = ET.SubElement(props, "Option", name=key, type="Map")
        ET.SubElement(p, "Option", name="active", type="bool", value="true")
        ET.SubElement(p, "Option", name="expression", type="QString", value=expression)
        ET.SubElement(p, "Option", name="type", type="int", value="3")
    ET.SubElement(root, "Option", name="type", type="QString", value="collection")


def _symbol(
    kind: str, name: str, layers: list[tuple[str, dict[str, Any], dict[str, str]]]
) -> ET.Element:
    sym = ET.Element(
        "symbol",
        type=kind,
        name=name,
        alpha="1",
        clip_to_extent="1",
        force_rhr="0",
        frame_rate="10",
        is_animated="0",
    )
    for cls, values, dd in layers:
        lay = ET.SubElement(
            sym, "layer", {"class": cls, "enabled": "1", "locked": "0", "pass": "0"}
        )
        _options(lay, values)
        _ddp(lay, dd)
    return sym


def _line(
    colour: str,
    width: float,
    *,
    style: str = "solid",
    alpha: float = 1.0,
    width_expr: str | None = None,
    offset: bool = True,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    values = {
        "line_color": _rgba(colour, alpha),
        "line_width": width,
        "line_width_unit": "MM",
        "line_style": style,
        "capstyle": "round",
        "joinstyle": "round",
        "use_custom_dash": "0",
        "draw_inside_polygon": "0",
        "offset": "0",
    }
    if style == "custom":
        values.update(
            line_style="solid", use_custom_dash="1", customdash="3;1.6", customdash_unit="MM"
        )
    dd = {"outlineWidth": width_expr} if width_expr else {}
    if offset:
        dd["offset"] = _LINE_OFFSET  # parallel circuits of one tower line side by side
    return ("SimpleLine", values, dd)


def _marker(
    shape: str,
    fill: str,
    size: float,
    *,
    outline: str = "#ffffff",
    outline_width: float = 0.3,
    size_expr: str | None = None,
    fill_alpha: float = 1.0,
    offset: str = "0,0",
    offset_expr: str | None = None,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    values = {
        "name": shape,
        "color": _rgba(fill, fill_alpha),
        "outline_color": _rgba(outline),
        "outline_style": "solid",
        "outline_width": outline_width,
        "outline_width_unit": "MM",
        "size": size,
        "size_unit": "MM",
        "angle": "0",
        "offset": offset,
        "offset_unit": "MM",
        "joinstyle": "miter",
        "scale_method": "diameter",
        "vertical_anchor_point": "1",
        "horizontal_anchor_point": "1",
    }
    dd = {"size": size_expr} if size_expr else {}
    if offset_expr:
        dd["offset"] = offset_expr
    return ("SimpleMarker", values, dd)


def _font_marker(
    char: str, size: float, colour: str = "#ffffff", size_expr: str | None = None
) -> tuple[str, dict[str, Any], dict[str, str]]:
    values = {
        "chr": char,
        "font": "Sans Serif",
        "font_style": "Bold",
        "color": _rgba(colour),
        "size": size,
        "size_unit": "MM",
        "outline_width": "0",
        "angle": "0",
        "offset": "0,0",
        "offset_unit": "MM",
        "vertical_anchor_point": "1",
        "horizontal_anchor_point": "1",
    }
    return ("FontMarker", values, {"size": size_expr} if size_expr else {})


def _fill(
    colour: str, alpha: float, outline: str, width: float, *, style: str = "solid"
) -> tuple[str, dict[str, Any], dict[str, str]]:
    values = {
        "color": _rgba(colour, alpha),
        "style": "solid" if alpha > 0 else "no",
        "outline_color": _rgba(outline),
        "outline_style": style,
        "outline_width": width,
        "outline_width_unit": "MM",
        "joinstyle": "round",
    }
    return ("SimpleFill", values, {})


def _quote(field: str) -> str:
    return f'"{field}"'


# --------------------------------------------------------------------------- rules
Rule = tuple[str, str, str, list[tuple[str, dict[str, Any], dict[str, str]]], int | None]
"""(label, filter expression, symbol type, symbol layers, max scale denominator)."""

_LINE_WIDTH = 'CASE WHEN "voltage_kv" >= 300 THEN 0.6 ELSE 0.4 END'
"""Width per circuit (mm); parallel circuits are drawn side by side (offset)."""
_LINE_OFFSET = '(coalesce("circuit_no", 0) - (coalesce("circuits", 1) - 1) / 2.0) * 0.8'
"""Offset (mm) of a circuit (piece) from the centre line of its tower line."""
TOWER_COLOUR = "#9e9e9e"
"""Physical tower lines (OpenStreetMap ``power=line``): thin grey casing under the circuits."""
_PLANT_SIZE = 'max(1.8, least(1.2 + 0.16 * sqrt(coalesce("capacity_mw", 0)), 8.0))'
_SUB_SIZE = (
    'CASE WHEN "max_kv" >= 300 THEN 2.6 ELSE 2.1 END'
    ' * (1 + 0.07 * least(coalesce("transformers", 0), 6))'
)
"""Substation symbol (mm): by highest voltage, growing with the transformer count."""


def _kv_colour(field: str) -> str:
    """QGIS expression: colour of the voltage class of ``field``."""
    return (
        f"CASE WHEN \"{field}\" >= 300 THEN '{_rgba(VOLTAGE_COLOURS['380'])}' "
        f"WHEN \"{field}\" >= 200 THEN '{_rgba(VOLTAGE_COLOURS['220'])}' "
        f"ELSE '{_rgba(VOLTAGE_COLOURS['110'])}' END"
    )


def _line_rules(mode: str) -> list[Rule]:
    rules: list[Rule] = [
        (
            "Out of service",
            "\"status\" <> 'in_operation'",
            "line",
            [_line(OUT_OF_SERVICE, 0.4, style="dot")],
            None,
        ),
        ("HVDC", '"hvdc" = 1', "line", [_line(VOLTAGE_COLOURS["hvdc"], 0.9, style="custom")], None),
    ]
    live = '"status" = \'in_operation\' AND "hvdc" = 0'
    if mode == "voltage":
        for kv, label in (("380", "380 kV"), ("220", "220 kV")):
            cond = '"voltage_kv" >= 300' if kv == "380" else '"voltage_kv" < 300'
            rules.append(
                (
                    f"{label} (route not mapped: dotted)",
                    f"{live} AND {cond} AND \"route_quality\" = 'straight'",
                    "line",
                    [_line(VOLTAGE_COLOURS[kv], 0.5, style="dot")],
                    None,
                )
            )
            rules.append(
                (
                    label,
                    f"{live} AND {cond} AND \"route_quality\" <> 'straight'",
                    "line",
                    [
                        _line("#ffffff", 1.1, alpha=0.9, width_expr=_LINE_WIDTH + " + 0.45"),
                        _line(VOLTAGE_COLOURS[kv], 0.7, width_expr=_LINE_WIDTH),
                    ],
                    None,
                )
            )
    else:
        rules.append(
            (
                "Interconnector (cross-border)",
                f'{live} AND "is_interconnector" = 1',
                "line",
                [
                    _line("#ffffff", 1.6, width_expr=_LINE_WIDTH + " + 0.9"),
                    _line(INTERCONNECTOR, 0.8, width_expr=_LINE_WIDTH),
                ],
                None,
            )
        )
        for tso in TSO_CODES:
            rules.append(
                (
                    tso,
                    f'{live} AND "is_interconnector" = 0 AND "tso" = \'{tso}\'',
                    "line",
                    [
                        _line("#ffffff", 1.1, alpha=0.9, width_expr=_LINE_WIDTH + " + 0.45"),
                        _line(TSO_COLOURS[tso], 0.7, width_expr=_LINE_WIDTH),
                    ],
                    None,
                )
            )
    return rules


def _tower_rules(mode: str) -> list[Rule]:
    return [
        (
            "Tower line (OpenStreetMap)",
            "",
            "line",
            [_line(TOWER_COLOUR, 0.35, alpha=0.85, offset=False)],
            _ZOOM_SCALE[9],
        )
    ]


def _substation_rules(mode: str) -> list[Rule]:
    ring = "\"symbol\" = 'tee'"
    rules: list[Rule] = [
        (
            "Tee / switching point",
            ring,
            "marker",
            [_marker("circle", "#ffffff", 1.3, outline=NEUTRAL, outline_width=0.35)],
            1_200_000,
        ),
    ]
    if mode == "voltage":
        rules += [
            (
                "Substation 380 kV",
                "\"symbol\" = '380'",
                "marker",
                [_marker("square", VOLTAGE_COLOURS["380"], 2.6, size_expr=_SUB_SIZE)],
                None,
            ),
            (
                "Substation 220 kV",
                "\"symbol\" = '220'",
                "marker",
                [
                    _marker(
                        "square",
                        "#ffffff",
                        2.1,
                        outline=VOLTAGE_COLOURS["220"],
                        outline_width=0.6,
                        size_expr=_SUB_SIZE,
                    )
                ],
                None,
            ),
            (
                "Coupling substation 380/220 kV",
                "\"symbol\" = 'coupling'",
                "marker",
                [
                    _marker("square", VOLTAGE_COLOURS["220"], 2.6, size_expr=_SUB_SIZE),
                    _marker(
                        "half_square",
                        VOLTAGE_COLOURS["380"],
                        2.6,
                        outline_width=0.0,
                        size_expr=_SUB_SIZE,
                    ),
                ],
                None,
            ),
        ]
    else:
        for tso in TSO_CODES:
            rules.append(
                (
                    tso,
                    f"NOT ({ring}) AND \"tso\" = '{tso}'",
                    "marker",
                    [
                        _marker(
                            "square",
                            TSO_COLOURS[tso],
                            2.2,
                            outline="#ffffff",
                            size_expr=_SUB_SIZE,
                        )
                    ],
                    None,
                )
            )
    return rules


def _plant_rules(mode: str) -> list[Rule]:
    rules: list[Rule] = [
        (
            "Not in operation / reserve",
            '"in_operation" = 0',
            "marker",
            [
                _marker(
                    "circle",
                    "#ffffff",
                    2.0,
                    outline=OUT_OF_SERVICE,
                    outline_width=0.3,
                    size_expr=_PLANT_SIZE,
                    fill_alpha=0.6,
                )
            ],
            1_200_000,
        ),
    ]
    if mode == "voltage":  # default mode: by fuel
        for cat, (label, shape, colour, letter) in FUEL_SYMBOLS.items():
            rules.append(
                (
                    label,
                    f'"in_operation" = 1 AND "display" = \'individual\' '
                    f"AND \"fuel_category\" = '{cat}'",
                    "marker",
                    [
                        _marker(shape, colour, 2.0, size_expr=_PLANT_SIZE),
                        _font_marker(letter, 1.4, size_expr=f"({_PLANT_SIZE}) * 0.55"),
                    ],
                    None,
                )
            )
    else:
        for tso in TSO_CODES:
            rules.append(
                (
                    tso,
                    f'"in_operation" = 1 AND "display" = \'individual\' AND "tso" = \'{tso}\'',
                    "marker",
                    [
                        _marker(
                            "circle", TSO_COLOURS[tso], 2.0, size_expr=_PLANT_SIZE, fill_alpha=0.8
                        )
                    ],
                    None,
                )
            )
        rules.append(
            (
                "Not attached",
                '"in_operation" = 1 AND "display" = \'individual\' AND "tso" IS NULL',
                "marker",
                [_marker("circle", OUT_OF_SERVICE, 1.6, size_expr=_PLANT_SIZE)],
                None,
            )
        )
    return rules


_TR_OFFSET = "format('%1,%2', 2.6 + 2.4 * coalesce(\"slot\", 0) + {dx}, -2.2)"
"""Transformer symbols in a row up-right of the substation (mm), one per transformer."""


def _transformer_rules(mode: str) -> list[Rule]:
    def circles(hv: str, lv: str) -> list[tuple[str, dict[str, Any], dict[str, str]]]:
        out = []
        for dx, colour in (("-0.5", hv), ("0.5", lv)):
            name, values, _ = _marker(
                "circle", "#ffffff", 1.9, outline=NEUTRAL, outline_width=0.4, fill_alpha=0.6
            )
            dd = {"offset": _TR_OFFSET.format(dx=dx)}
            if colour.startswith("CASE"):
                dd["outlineColor"] = colour
            else:
                values["outline_color"] = _rgba(colour)
            out.append((name, values, dd))
        return out

    rules: list[Rule] = [
        (
            "Phase-shifting transformer",
            '"phase_shifter" = 1',
            "marker",
            [
                _marker(
                    "diamond",
                    INTERCONNECTOR,
                    2.6,
                    outline="#ffffff",
                    offset_expr=_TR_OFFSET.format(dx="0"),
                )
            ],
            _ZOOM_SCALE[7],
        ),
    ]
    if mode == "voltage":
        rules.append(
            (
                "Transformer (HV circle left, LV circle right; colour = voltage)",
                '"phase_shifter" = 0',
                "marker",
                circles(_kv_colour("kv_hv"), _kv_colour("kv_lv")),
                _ZOOM_SCALE[7],
            )
        )
    else:
        for tso in TSO_CODES:
            rules.append(
                (
                    tso,
                    f'"phase_shifter" = 0 AND "tso" = \'{tso}\'',
                    "marker",
                    circles(TSO_COLOURS[tso], TSO_COLOURS[tso]),
                    _ZOOM_SCALE[7],
                )
            )
    return rules


def _plant_connection_rules(mode: str) -> list[Rule]:
    return [
        (
            "Plant connection (to its substation)",
            '"in_operation" = 1',
            "line",
            [("SimpleLine", {**_line(NEUTRAL, 0.25, alpha=0.7)[1]}, {})],
            _ZOOM_SCALE[7],
        )
    ]


_AGG_OFFSET = "format('%1,%2', -3.0 - 2.4 * coalesce(\"slot\", 0), 0)"


def _plant_aggregate_rules(mode: str) -> list[Rule]:
    rules: list[Rule] = []
    for cat, (label, shape, colour, letter) in FUEL_SYMBOLS.items():
        sym = _marker(shape, colour, 2.0)
        txt = _font_marker(letter, 1.2)
        rules.append(
            (
                f"{label} (< {PLANT_INDIVIDUAL_MW:g} MW units, summed per bus)",
                f"\"fuel_category\" = '{cat}'",
                "marker",
                [
                    (sym[0], sym[1], {"offset": _AGG_OFFSET}),
                    (txt[0], txt[1], {"offset": _AGG_OFFSET}),
                ],
                _ZOOM_SCALE[8],
            )
        )
    return rules


def _bus_rules(mode: str) -> list[Rule]:
    rules: list[Rule] = [
        (
            "Connection / terminal bus (110/21 kV, at its substation)",
            "\"kind\" IN ('connection', 'terminal')",
            "marker",
            [_marker("circle", "#bdbdbd", 0.9, outline=NEUTRAL, outline_width=0.15)],
            600_000,
        ),
    ]
    other = "\"kind\" NOT IN ('connection', 'terminal')"
    if mode == "voltage":
        rules += [
            (
                "Bus 380 kV",
                f'{other} AND "voltage_kv" >= 300',
                "marker",
                [
                    _marker(
                        "circle", VOLTAGE_COLOURS["380"], 1.1, outline="#ffffff", outline_width=0.15
                    )
                ],
                None,
            ),
            (
                "Bus 220 kV",
                f'{other} AND "voltage_kv" < 300',
                "marker",
                [
                    _marker(
                        "circle", VOLTAGE_COLOURS["220"], 1.1, outline="#ffffff", outline_width=0.15
                    )
                ],
                None,
            ),
        ]
    else:
        for tso in TSO_CODES:
            rules.append(
                (
                    tso,
                    f"{other} AND \"tso\" = '{tso}'",
                    "marker",
                    [
                        _marker(
                            "circle", TSO_COLOURS[tso], 1.1, outline="#ffffff", outline_width=0.15
                        )
                    ],
                    None,
                )
            )
    return rules


def _neighbour_rules(mode: str) -> list[Rule]:
    return [
        (
            "Neighbouring network (equivalent)",
            "",
            "marker",
            [
                _marker("circle", NEUTRAL, 2.6, outline="#ffffff", outline_width=0.4),
                _marker("cross2", "#ffffff", 1.4, outline="#ffffff", outline_width=0.3),
            ],
            None,
        )
    ]


def _control_rules(mode: str) -> list[Rule]:
    rules: list[Rule] = []
    for tso in TSO_CODES:
        if mode == "voltage":
            sym = [_fill(TSO_COLOURS[tso], 0.0, TSO_COLOURS[tso], 0.35, style="dash")]
        else:
            sym = [_fill(TSO_COLOURS[tso], 0.16, TSO_COLOURS[tso], 0.6)]
        rules.append((f"{tso} (approximate)", f"\"tso\" = '{tso}'", "fill", sym, None))
    return rules


def _zone_rules(mode: str) -> list[Rule]:
    return [
        (
            "DE-LU bidding zone",
            "\"zone\" = 'DE-LU'",
            "fill",
            [_fill("#ffffff", 0.0, "#212121", 0.9 if mode == "tso" else 0.5)],
            None,
        ),
        (
            "Neighbouring bidding zone",
            "\"zone\" <> 'DE-LU'",
            "fill",
            [_fill("#eceff1", 0.45, "#757575", 0.3, style="dash")],
            None,
        ),
    ]


# --------------------------------------------------------------------------- labels
Label = tuple[str, str, int, float, str, str]
"""(description, field expression, max scale denominator, font size pt, placement, filter)."""

_LABELS: dict[str, list[Label]] = {
    "lines": [
        ("Line name (one per circuit)", '"name"', _ZOOM_SCALE[10], 7.0, "line", '"label_piece"'),
        (
            "Circuits on the tower line",
            "\"circuits\" || '×'",
            _ZOOM_SCALE[7],
            7.0,
            "line",
            '"circuit_no" = 0 AND "circuits" >= 2 AND @map_scale > ' + str(_ZOOM_SCALE[10]),
        ),
    ],
    "tower_lines": [
        ("Tower line ref", '"ref"', _ZOOM_SCALE[11] // 2, 6.0, "line", "\"ref\" <> ''")
    ],
    "substations": [
        (
            "380 kV substation names",
            '"name"',
            _ZOOM_SCALE[8],
            8.0,
            "point",
            '"max_kv" >= 300 AND "kind" = \'substation\'',
        ),
        (
            "220 kV substation names",
            '"name"',
            _ZOOM_SCALE[9],
            7.5,
            "point",
            '"max_kv" < 300 AND "kind" = \'substation\'',
        ),
    ],
    "plants": [
        (
            "Plant name and MW",
            "\"name\" || ' (' || round(\"capacity_mw\") || ' MW)'",
            _ZOOM_SCALE[8],
            7.0,
            "point",
            f'"in_operation" = 1 AND "capacity_mw" >= {PLANT_LABEL_MW}',
        )
    ],
    "plant_connections": [],
    "plant_aggregates": [
        ("Summed MW", "round(\"capacity_mw\") || ' MW'", _ZOOM_SCALE[11], 6.0, "point", "")
    ],
    "transformers": [],
    "buses": [("Bus id", '"bus"', _ZOOM_SCALE[11], 6.5, "point", "")],
    "neighbours": [
        (
            "Neighbour zone",
            "\"name\" || ' (' || \"bidding_zone\" || ')'",
            _ZOOM_SCALE[6],
            8.0,
            "point",
            "",
        )
    ],
    "control_areas": [("Control area", '"tso"', _ZOOM_SCALE[6] * 2, 11.0, "polygon", "")],
    "bidding_zones": [("Bidding zone", '"zone"', _ZOOM_SCALE[6] * 2, 10.0, "polygon", "")],
}

_RULES = {
    "lines": _line_rules,
    "tower_lines": _tower_rules,
    "plant_connections": _plant_connection_rules,
    "plant_aggregates": _plant_aggregate_rules,
    "substations": _substation_rules,
    "plants": _plant_rules,
    "transformers": _transformer_rules,
    "buses": _bus_rules,
    "neighbours": _neighbour_rules,
    "control_areas": _control_rules,
    "bidding_zones": _zone_rules,
}
STYLED_LAYERS = tuple(_RULES)
MODES = {"voltage": "By voltage", "tso": "By TSO"}


def _labeling(parent: ET.Element, labels: list[Label]) -> None:
    if not labels:
        return
    lab = ET.SubElement(parent, "labeling", type="rule-based")
    rules = ET.SubElement(lab, "rules", key="{labels-root}")
    for i, (desc, expr, max_scale, size, placement, flt) in enumerate(labels):
        rule = ET.SubElement(
            rules,
            "rule",
            key=f"{{label-{i}}}",
            description=desc,
            scalemaxdenom=str(max_scale),
            scalemindenom="1",
        )
        if flt:
            rule.set("filter", flt)
        settings = ET.SubElement(rule, "settings", calloutType="simple")
        style = ET.SubElement(
            settings,
            "text-style",
            fieldName=expr,
            isExpression="1",
            fontFamily="Sans Serif",
            fontSize=str(size),
            fontSizeUnit="Point",
            fontWeight="50",
            textColor="33,33,33,255",
            textOpacity="1",
            fontItalic="0" if placement != "polygon" else "1",
            namedStyle="Regular",
            multilineHeight="1",
            allowHtml="0",
        )
        ET.SubElement(
            style,
            "text-buffer",
            bufferDraw="1",
            bufferSize="0.8",
            bufferSizeUnits="MM",
            bufferColor="255,255,255,230",
            bufferOpacity="0.9",
            bufferJoinStyle="128",
            bufferNoFill="1",
        )
        ET.SubElement(
            settings,
            "text-format",
            wrapChar="",
            autoWrapLength="0",
            multilineAlign="0",
            plussign="0",
            reverseDirectionSymbol="0",
        )
        code = {"point": "6", "line": "2", "polygon": "1"}[placement]
        ET.SubElement(
            settings,
            "placement",
            placement=code,
            placementFlags="10",
            dist="1.2",
            distUnits="MM",
            priority="5",
            repeatDistance="0",
            overrunDistance="0",
            offsetType="0",
            quadOffset="4",
            centroidInside="1",
            fitInPolygonOnly="0",
        )
        ET.SubElement(
            settings,
            "rendering",
            obstacle="1",
            obstacleFactor="1",
            obstacleType="1",
            upsidedownLabels="0",
            displayAll="0",
            mergeLines="1" if placement == "line" else "0",
            labelPerPart="0",
            limitNumLabels="0",
            maxNumLabels="2000",
            fontLimitPixelSize="0",
            scaleVisibility="0",
            drawLabels="1",
        )


def layer_style(layer: str, mode: str) -> ET.Element:
    """``<qgis>`` element of the style of ``layer`` in ``mode`` ("voltage" or "tso")."""
    root = ET.Element("qgis", version=QGIS_VERSION, styleCategories="Symbology|Labeling")
    renderer = ET.SubElement(
        root,
        "renderer-v2",
        type="RuleRenderer",
        symbollevels="0",
        enableorderby="0",
        forceraster="0",
        referencescale="-1",
    )
    rules = ET.SubElement(renderer, "rules", key="{root}")
    symbols = ET.SubElement(renderer, "symbols")
    for i, (label, flt, kind, layers, max_scale) in enumerate(_RULES[layer](mode)):
        rule = ET.SubElement(rules, "rule", key=f"{{r{i}}}", label=label, symbol=str(i))
        if flt:
            rule.set("filter", flt)
        if max_scale:
            rule.set("scalemaxdenom", str(max_scale))
            rule.set("scalemindenom", "1")
        symbols.append(_symbol(kind, str(i), layers))
    _labeling(root, _LABELS[layer])
    ET.SubElement(root, "layerGeometryType").text = {
        "lines": "1",
        "tower_lines": "1",
        "plant_connections": "1",
        "control_areas": "2",
        "bidding_zones": "2",
    }.get(layer, "0")
    return root


def _serialise(root: ET.Element) -> str:
    ET.indent(root)
    return (
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        + ET.tostring(root, encoding="unicode")
        + "\n"
    )


def write_qgis_styles(out_dir: str | Path) -> list[Path]:
    """``<layer>_<mode>.qml`` for every styled layer and mode."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for layer in STYLED_LAYERS:
        for mode in MODES:
            path = out / f"{layer}_{mode}.qml"
            path.write_text(_serialise(layer_style(layer, mode)), encoding="utf-8")
            paths.append(path)
    return paths


# --------------------------------------------------------------------------- project
_GEOMETRY = {
    "lines": "Line",
    "tower_lines": "Line",
    "plant_connections": "Line",
    "control_areas": "Polygon",
    "bidding_zones": "Polygon",
}
_TITLES = {
    "lines": "Lines (AC circuits, HVDC)",
    "tower_lines": "Tower lines (OpenStreetMap)",
    "substations": "Substations",
    "transformers": "Transformers",
    "plants": "Power plants (MaStR)",
    "plant_aggregates": "Small plants (summed per bus and fuel)",
    "plant_connections": "Plant connections",
    "buses": "Buses (model / ANDES ids)",
    "neighbours": "Neighbouring networks",
    "control_areas": "TSO control areas (approximate)",
    "bidding_zones": "Bidding zones",
}
_VISIBLE = {
    "lines",
    "tower_lines",
    "substations",
    "plants",
    "plant_aggregates",
    "plant_connections",
    "neighbours",
    "bidding_zones",
    "control_areas",
    "transformers",
}
_ORDER = (
    "neighbours",
    "plants",
    "plant_aggregates",
    "transformers",
    "substations",
    "buses",
    "lines",
    "tower_lines",
    "plant_connections",
    "control_areas",
    "bidding_zones",
)
"""Layer tree order, top first."""


def _srs(parent: ET.Element, epsg: int) -> None:
    srs = ET.SubElement(parent, "spatialrefsys", nativeFormat="Wkt")
    proj = {
        4326: "+proj=longlat +datum=WGS84 +no_defs",
        3857: "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 "
        "+units=m +nadgrids=@null +wktext +no_defs",
    }[epsg]
    for tag, text in (
        ("proj4", proj),
        ("srid", str(epsg)),
        ("authid", f"EPSG:{epsg}"),
        ("description", "WGS 84" if epsg == 4326 else "WGS 84 / Pseudo-Mercator"),
        ("projectionacronym", "longlat" if epsg == 4326 else "merc"),
        ("ellipsoidacronym", "EPSG:7030"),
        ("geographicflag", "true" if epsg == 4326 else "false"),
    ):
        ET.SubElement(srs, tag).text = text


def write_qgis_project(
    out_dir: str | Path,
    *,
    gpkg_name: str = "grid_model.gpkg",
    title: str = "German transmission grid model",
) -> Path:
    """``grid_model.qgs`` (relative paths: put ``gpkg_name`` in the same folder)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = ET.Element("qgis", projectname=title, version=QGIS_VERSION)
    ET.SubElement(root, "homePath", path="")
    ET.SubElement(root, "title").text = title
    crs = ET.SubElement(root, "projectCrs")
    _srs(crs, 3857)
    tree = ET.SubElement(root, "layer-tree-group")
    ET.SubElement(tree, "customproperties")
    ids = {name: f"{name}_grid_model" for name in STYLED_LAYERS}
    ids["basemap"] = "basemap_carto_positron"
    for name in (*_ORDER, "basemap"):
        ET.SubElement(
            tree,
            "layer-tree-layer",
            id=ids[name],
            name=_TITLES.get(name, BASEMAP["name"]),
            source=f"./{gpkg_name}|layername={name}" if name != "basemap" else "",
            providerKey="ogr" if name != "basemap" else "wms",
            checked="Qt::Checked" if name in _VISIBLE or name == "basemap" else "Qt::Unchecked",
            expanded="0",
        )
    layers = ET.SubElement(root, "projectlayers")
    for name in STYLED_LAYERS:
        styles = {mode: layer_style(name, mode) for mode in MODES}
        ml = ET.SubElement(
            layers,
            "maplayer",
            type="vector",
            geometry=_GEOMETRY.get(name, "Point"),
            hasScaleBasedVisibilityFlag="0",
            minScale="100000000",
            maxScale="0",
            labelsEnabled="1" if _LABELS[name] else "0",
            readOnly="0",
        )
        ET.SubElement(ml, "id").text = ids[name]
        ET.SubElement(ml, "datasource").text = f"./{gpkg_name}|layername={name}"
        ET.SubElement(ml, "layername").text = _TITLES[name]
        srs = ET.SubElement(ml, "srs")
        _srs(srs, 4326)
        ET.SubElement(ml, "provider", encoding="UTF-8").text = "ogr"
        for child in styles["voltage"]:
            if child.tag != "layerGeometryType":
                ml.append(child)
        if name in ("control_areas", "bidding_zones"):
            ET.SubElement(ml, "layerOpacity").text = "1"
        manager = ET.SubElement(ml, "map-layer-style-manager", current=MODES["voltage"])
        for mode, label in MODES.items():
            entry = ET.SubElement(manager, "map-layer-style", name=label)
            entry.append(styles[mode])
    base = ET.SubElement(layers, "maplayer", type="raster", hasScaleBasedVisibilityFlag="0")
    ET.SubElement(base, "id").text = ids["basemap"]
    url = BASEMAP["url"].replace("{", "%7B").replace("}", "%7D")
    ET.SubElement(
        base, "datasource"
    ).text = f"type=xyz&url={url}&zmax={BASEMAP['zmax']}&zmin=0&http-header:referer="
    ET.SubElement(base, "layername").text = BASEMAP["name"]
    srs = ET.SubElement(base, "srs")
    _srs(srs, 3857)
    ET.SubElement(base, "provider").text = "wms"
    ET.SubElement(base, "attribution").text = BASEMAP["attribution"]
    ET.SubElement(base, "layerOpacity").text = "0.85"
    canvas = ET.SubElement(root, "mapcanvas", name="theMapCanvas", annotationsVisible="1")
    ET.SubElement(canvas, "units").text = "meters"
    extent = ET.SubElement(canvas, "extent")
    for tag, value in (("xmin", 580000), ("ymin", 5950000), ("xmax", 1720000), ("ymax", 7450000)):
        ET.SubElement(extent, tag).text = str(value)
    ET.SubElement(canvas, "rotation").text = "0"
    dest = ET.SubElement(canvas, "destinationsrs")
    _srs(dest, 3857)
    props = ET.SubElement(root, "properties")
    paths = ET.SubElement(props, "Paths")
    ET.SubElement(paths, "Absolute", type="bool").text = "false"
    meta = ET.SubElement(root, "projectMetadata")
    ET.SubElement(meta, "title").text = title
    ET.SubElement(meta, "abstract").text = (
        "Frozen grid model v4final (380/220 kV). Route geometry © OpenStreetMap contributors "
        "(ODbL) via PyPSA-Eur; borders Natural Earth (public domain); electrical data JAO "
        "Core Static Grid Model; plants Marktstammdatenregister (dl-de/by-2-0). Basemap "
        + BASEMAP["attribution"]
        + "."
    )
    path = out / "grid_model.qgs"
    path.write_text(_serialise(root), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- web/SVG
def _svg_shape(shape: str, fill: str, stroke: str = "#ffffff", sw: float = 1.5) -> str:
    paint = f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
    geometry = {
        "circle": '<circle cx="12" cy="12" r="10" {}/>',
        "square": '<rect x="2.5" y="2.5" width="19" height="19" {}/>',
        "triangle": '<polygon points="12,1.5 22.5,21.5 1.5,21.5" {}/>',
        "diamond": '<polygon points="12,1 23,12 12,23 1,12" {}/>',
        "hexagon": '<polygon points="12,1.5 21.5,7 21.5,17 12,22.5 2.5,17 2.5,7" {}/>',
    }
    if shape in geometry:
        return geometry[shape].format(paint)
    raise ValueError(shape)


def write_svg_symbols(out_dir: str | Path) -> list[Path]:
    """SVG symbols (24 x 24) of plants by fuel, substations and transformers."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    head = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">'
    files = {}
    for cat, (_, shape, colour, letter) in FUEL_SYMBOLS.items():
        ty = "16.5" if shape == "triangle" else "16"
        files[f"plant_{cat}.svg"] = (
            head + _svg_shape(shape, colour) + f'<text x="12" y="{ty}" font-family="sans-serif" '
            f'font-size="{9 if shape == "triangle" else 11}" font-weight="700" fill="#fff" '
            f'text-anchor="middle">{letter}</text></svg>'
        )
    files["substation_380.svg"] = head + _svg_shape("square", VOLTAGE_COLOURS["380"]) + "</svg>"
    files["substation_220.svg"] = (
        head + _svg_shape("square", "#ffffff", VOLTAGE_COLOURS["220"], 3.0) + "</svg>"
    )
    files["substation_coupling.svg"] = (
        head + f'<rect x="2.5" y="2.5" width="19" height="19" fill="{VOLTAGE_COLOURS["220"]}"/>'
        f'<polygon points="2.5,2.5 21.5,2.5 2.5,21.5" fill="{VOLTAGE_COLOURS["380"]}"/>'
        '<rect x="2.5" y="2.5" width="19" height="19" fill="none" stroke="#fff" '
        'stroke-width="1.5"/></svg>'
    )
    for tso in TSO_CODES:
        files[f"substation_{tso.lower()}.svg"] = (
            head + _svg_shape("square", TSO_COLOURS[tso]) + "</svg>"
        )
    files["tee.svg"] = (
        head
        + f'<circle cx="12" cy="12" r="7" fill="#fff" stroke="{NEUTRAL}" stroke-width="2.5"/></svg>'
    )
    kv_colour = {"380": VOLTAGE_COLOURS["380"], "220": VOLTAGE_COLOURS["220"], "110": NEUTRAL}
    for hv in KV_CLASSES:
        for lv in KV_CLASSES:
            files[f"transformer_{hv}_{lv}.svg"] = (
                head + '<circle cx="8.3" cy="12" r="6.6" fill="#fff" fill-opacity="0.85" '
                'stroke="#fff" stroke-width="4.4"/>'
                '<circle cx="15.7" cy="12" r="6.6" fill="#fff" fill-opacity="0.85" '
                'stroke="#fff" stroke-width="4.4"/>'
                f'<circle cx="8.3" cy="12" r="6.6" fill="none" stroke="{kv_colour[hv]}" '
                'stroke-width="2.4"/>'
                f'<circle cx="15.7" cy="12" r="6.6" fill="none" stroke="{kv_colour[lv]}" '
                'stroke-width="2.4"/></svg>'
            )
    files["transformer.svg"] = files["transformer_380_220.svg"]
    files["phase_shifter.svg"] = head + _svg_shape("diamond", INTERCONNECTOR) + "</svg>"
    files["neighbour.svg"] = (
        head + f'<circle cx="12" cy="12" r="10" fill="{NEUTRAL}" stroke="#fff" stroke-width="1.5"/>'
        '<path d="M7 7 L17 17 M17 7 L7 17" stroke="#fff" stroke-width="2.2"/></svg>'
    )
    paths = []
    for name, text in files.items():
        path = out / name
        path.write_text(text + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def web_style() -> dict[str, Any]:
    """Colours, symbols and zoom thresholds for the web viewer (same spec as QGIS)."""
    return {
        "voltage": VOLTAGE_COLOURS,
        "tso": TSO_COLOURS,
        "out_of_service": OUT_OF_SERVICE,
        "interconnector": INTERCONNECTOR,
        "neutral": NEUTRAL,
        "fuels": {
            cat: {"label": label, "shape": shape, "colour": colour, "letter": letter}
            for cat, (label, shape, colour, letter) in FUEL_SYMBOLS.items()
        },
        "fuel_sources": FUEL_CATEGORY,
        "label_zoom": {
            "substation_380": 8,
            "substation_220": 9,
            "corridor": 7,
            "line": 10.5,
            "plant": 8,
            "plant_aggregate": 8.5,
            "plant_aggregate_mw": 11.5,
            "transformer": 7.5,
            "transformer_all": 11,
        },
        "lines": {"zoom": [list(z) for z in LINE_ZOOM]},
        "plants": {
            "size_stops": [list(s) for s in PLANT_SIZE_STOPS],
            "zoom_factor": [[5, 0.6], [8, 0.8], [12, 1.05]],
            "label_mw": PLANT_LABEL_MW,
            "individual_mw": PLANT_INDIVIDUAL_MW,
        },
        "substations": {"zoom_size": [[5, 0.32], [8, 0.52], [12, 0.85], [15, 1.05]]},
        "transformer_symbols": [f"transformer_{h}_{lv}" for h in KV_CLASSES for lv in KV_CLASSES],
        "basemap_attribution": BASEMAP["attribution"],
    }


def write_all(app_dir: str | Path) -> dict[str, Any]:
    """Write QGIS styles + project (``qgis/``), SVG symbols and ``style.js`` into ``app_dir``."""
    app = Path(app_dir)
    styles = write_qgis_styles(app / "qgis" / "styles")
    project = write_qgis_project(app / "qgis")
    symbols = write_svg_symbols(app / "symbols")
    js = app / "style.js"
    js.write_text(
        "// Generated by tranpy.grid_gis_style.write_all - do not edit by hand.\n"
        f"window.GRID_STYLE = {json.dumps(web_style(), indent=1, ensure_ascii=False)};\n",
        encoding="utf-8",
    )
    return {"styles": styles, "project": project, "symbols": symbols, "style_js": js}
