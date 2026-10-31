"""Which CGMES tables can be filled, and from what. The classes come from the schema."""

from __future__ import annotations

import pandas as pd

from tranpy.cim.schema import tables

FILL = 1
LATER = 2

SOURCES: dict[str, tuple[int, str, str]] = {
    # network
    "GeographicalRegion": (FILL, "derived", "Germany and neighbours"),
    "SubGeographicalRegion": (FILL, "grid_de_nodes", "control area"),
    "BaseVoltage": (FILL, "grid_de_nodes", "220, 380 kV and foreign levels"),
    "Substation": (FILL, "grid_de_nodes", "site"),
    "VoltageLevel": (FILL, "grid_de_nodes", "site and voltage"),
    "BusbarSection": (FILL, "grid_de_nodes", "one per voltage level"),
    "ConnectivityNode": (FILL, "grid_de_nodes", "one per voltage level"),
    "TopologicalNode": (FILL, "grid_de_nodes", "one per voltage level"),
    "Terminal": (FILL, "derived", "one per equipment end"),
    "Line": (FILL, "grid_de_circuits", "container per circuit"),
    "ACLineSegment": (FILL, "grid_de_circuits", "JAO parameters"),
    "PowerTransformer": (FILL, "grid_de_transformers", "JAO"),
    "PowerTransformerEnd": (FILL, "grid_de_transformers", "JAO"),
    "PhaseTapChangerLinear": (LATER, "JAO", "angle dropped in cleaning, restore first"),
    "OperationalLimitSet": (FILL, "grid_de_circuits", "per terminal"),
    "OperationalLimitType": (FILL, "derived", "permanent limit"),
    "CurrentLimit": (FILL, "grid_de_circuits", "JAO rating"),
    "VoltageLimit": (LATER, "none yet", "needs a source"),
    # injections
    "SynchronousMachine": (FILL, "grid_de_plant_attachments", "conventional, hydro"),
    "ThermalGeneratingUnit": (FILL, "grid_de_plant_attachments", "coal, gas, oil, waste"),
    "HydroGeneratingUnit": (FILL, "grid_de_plant_attachments", "hydro, pumped"),
    "HydroPowerPlant": (LATER, "plant list", "pumped storage"),
    "HydroPump": (LATER, "plant list", "pumped storage"),
    "NuclearGeneratingUnit": (LATER, "plant list", "only if in operation or reserve"),
    "PowerElectronicsConnection": (FILL, "grid_de_register_cells", "per node and technology"),
    "PhotoVoltaicUnit": (FILL, "grid_de_register_cells", "aggregated"),
    "PowerElectronicsWindUnit": (FILL, "grid_de_register_cells", "aggregated"),
    "BatteryUnit": (FILL, "grid_de_register_cells", "aggregated; energy needs the register"),
    "WindGeneratingUnit": (LATER, "register", "older machine-based wind"),
    "AsynchronousMachine": (LATER, "register", "older machine-based wind"),
    "ConformLoad": (FILL, "grid_de_load_shares", "SMARD area split by population"),
    "LoadArea": (FILL, "grid_de_area_load", "control area"),
    "SubLoadArea": (FILL, "grid_de_area_load", "control area"),
    "ConformLoadGroup": (FILL, "grid_de_area_load", "control area"),
    "ExternalNetworkInjection": (FILL, "grid_de_nodes", "foreign nodes"),
    "ControlArea": (LATER, "SMARD", "needs scenario data"),
    "TieFlow": (LATER, "grid_de_circuits", "needs ControlArea"),
    # hvdc, compensation, switching
    "DCLineSegment": (LATER, "JAO, OSM, public pages", "parameters to source"),
    "VsConverter": (LATER, "public pages", "ratings to source"),
    "CsConverter": (LATER, "public pages", "ratings to source"),
    "LinearShuntCompensator": (LATER, "grid_de_osm_*", "location only, no ratings"),
    "StaticVarCompensator": (LATER, "public pages", "ratings to source"),
    "Breaker": (LATER, "grid_de_osm_switches", "no state known; not mapped to terminals"),
    "Disconnector": (LATER, "grid_de_osm_switches", "no state known; not mapped to terminals"),
    # geometry, header
    "CoordinateSystem": (FILL, "derived", "WGS84"),
    "Location": (FILL, "grid_de_circuits, grid_de_nodes", "one per object"),
    "PositionPoint": (FILL, "grid_de_circuits, grid_de_nodes", "route points"),
    "FullModel": (FILL, "derived", "one per profile"),
    # state
    "SvVoltage": (LATER, "power flow", "per scenario"),
    "SvPowerFlow": (LATER, "power flow", "per scenario"),
}


def report() -> pd.DataFrame:
    """One row per table: profile, column counts, plan, source."""
    rows = []
    for table in tables():
        plan, source, note = SOURCES.get(table.name, (0, "", ""))
        mandatory = [c.name for c in table.columns if c.multiplicity.startswith("1")]
        rows.append(
            {
                "table": table.name.lower(),
                "cim_class": table.name,
                "profiles": ", ".join(table.profiles),
                "columns": len(table.columns),
                "mandatory_columns": len(mandatory),
                "plan": {FILL: "fill now", LATER: "later", 0: "empty"}[plan],
                "source": source,
                "note": note,
            }
        )
    return pd.DataFrame(rows)
