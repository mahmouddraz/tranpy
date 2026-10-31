from __future__ import annotations

from tranpy.cim.schema import create_statement, tables


def _by_name() -> dict:
    return {t.name: t for t in tables()}


def test_every_table_is_a_concrete_class_with_unique_columns() -> None:
    all_tables = tables()
    assert len(all_tables) > 300
    assert len({t.name.lower() for t in all_tables}) == len(all_tables)
    for table in all_tables:
        names = [c.name for c in table.columns]
        assert len(names) == len(set(names)), table.name


def test_inverter_and_battery_classes_exist() -> None:
    t = _by_name()
    for name in ("PowerElectronicsConnection", "PhotoVoltaicUnit", "PowerElectronicsWindUnit"):
        assert name in t
    columns = {c.name: c for c in t["BatteryUnit"].columns}
    assert {"BatteryUnit.ratedE", "BatteryUnit.storedE", "BatteryUnit.batteryState"} <= set(columns)
    assert "IdentifiedObject.mRID" in columns
    assert columns["BatteryUnit.batteryState"].sql_type == "STRING"
    assert "PowerElectronicsUnit" in t["BatteryUnit"].ancestors


def test_columns_are_inherited_and_typed() -> None:
    columns = {c.name: c for c in _by_name()["ACLineSegment"].columns}
    assert columns["ACLineSegment.r"].sql_type == "DOUBLE"
    assert columns["ACLineSegment.r"].multiplicity == "1..1"
    assert columns["IdentifiedObject.name"].sql_type == "STRING"
    assert "Conductor.length" in columns
    assert "ConductingEquipment.BaseVoltage" in columns


def test_abstract_classes_and_datatypes_have_no_table() -> None:
    t = _by_name()
    for name in ("IdentifiedObject", "ConductingEquipment", "Resistance", "Float"):
        assert name not in t


def test_create_statement_keeps_exact_names() -> None:
    sql = create_statement(_by_name()["ACLineSegment"], "c.cgmes_v3_0")
    assert "c.cgmes_v3_0.aclinesegment" in sql
    assert "`ACLineSegment.r` DOUBLE" in sql
    assert "CGMES 3.0 class ACLineSegment" in sql
