"""Parse the published CGMES 3.0 RDFS files into tables and columns.

Every concrete class is one table; every attribute or association of the class and its
ancestors, in any profile, is one column. Nothing is added or removed. A column has the short
name of the attribute (``r``), since the table already gives the class; the published name
(``ACLineSegment.r``) is its comment. Unity Catalog stores table names in lower case, so the
exact class name is the table comment.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from importlib import resources

_RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
_RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
_CIMS = "{http://iec.ch/TC57/1999/rdf-schema-extensions-19990926#}"

_PRIMITIVES = {
    "Boolean": "BOOLEAN",
    "Integer": "BIGINT",
    "Float": "DOUBLE",
    "Decimal": "DOUBLE",
    "String": "STRING",
    "Date": "STRING",
    "DateTime": "STRING",
    "Duration": "STRING",
    "MonthDay": "STRING",
    "Time": "STRING",
    "URI": "STRING",
    "IRI": "STRING",
    "UUID": "STRING",
    "StringFixedLanguage": "STRING",
    "StringIRI": "STRING",
    "Version": "STRING",
}
"""SQL type of the primitive types. Dates stay text: the standard allows partial forms."""


@dataclass(frozen=True)
class Column:
    """One published attribute or association."""

    cim_name: str
    """Published name, ``Class.attribute``."""
    sql_type: str
    multiplicity: str
    profiles: tuple[str, ...]
    range: str | None = None
    """Target class of an association, or enumeration of an attribute."""

    @property
    def name(self) -> str:
        """Column name: the attribute without its class."""
        return self.cim_name.rsplit(".", 1)[-1]


@dataclass
class Table:
    """One concrete class."""

    name: str
    profiles: tuple[str, ...]
    ancestors: tuple[str, ...]
    columns: list[Column] = field(default_factory=list)


@dataclass
class _Class:
    profiles: set[str] = field(default_factory=set)
    parent: str | None = None
    stereotypes: set[str] = field(default_factory=set)


@dataclass
class _Property:
    name: str
    domain: str
    multiplicity: str
    range: str | None
    data_type: str | None
    profiles: set[str] = field(default_factory=set)


def _local(value: str | None) -> str:
    return (value or "").split("#")[-1]


def _resource(element: ET.Element | None) -> str | None:
    return None if element is None else _local(element.get(f"{_RDF}resource"))


def _read() -> tuple[dict[str, _Class], dict[tuple[str, str], _Property]]:
    classes: dict[str, _Class] = {}
    properties: dict[tuple[str, str], _Property] = {}
    folder = resources.files("tranpy.resources.cgmes")
    for path in sorted(folder.iterdir(), key=lambda p: p.name):
        if not path.name.endswith("RDFS2020.rdf"):
            continue
        profile = path.name.split("600-2_")[1].split("-AP")[0]
        with path.open("rb") as fh:
            root = ET.parse(fh).getroot()
        for item in root.findall(f"{_RDF}Description"):
            name = _local(item.get(f"{_RDF}about"))
            kinds = {_local(t.get(f"{_RDF}resource")) for t in item.findall(f"{_RDF}type")}
            if "Class" in kinds:
                entry = classes.setdefault(name, _Class())
                entry.profiles.add(profile)
                entry.stereotypes |= {
                    _local(s.get(f"{_RDF}resource") or s.text)
                    for s in item.findall(f"{_CIMS}stereotype")
                }
                entry.parent = _resource(item.find(f"{_RDFS}subClassOf")) or entry.parent
            elif "Property" in kinds:
                domain = _resource(item.find(f"{_RDFS}domain"))
                if domain is None:
                    continue
                prop = properties.setdefault(
                    (domain, name),
                    _Property(
                        name,
                        domain,
                        (_resource(item.find(f"{_CIMS}multiplicity")) or "").removeprefix("M:"),
                        _resource(item.find(f"{_RDFS}range")),
                        _resource(item.find(f"{_CIMS}dataType")),
                    ),
                )
                prop.profiles.add(profile)
    return classes, properties


def _sql_type(prop: _Property, properties: dict[tuple[str, str], _Property]) -> str:
    many = prop.multiplicity.endswith("n") or prop.multiplicity == "0..2"
    base = "STRING"  # associations and enumerations hold a reference
    if prop.data_type is not None:
        data_type = prop.data_type
        value = properties.get((data_type, f"{data_type}.value"))
        if value is not None and value.data_type:
            data_type = value.data_type
        base = _PRIMITIVES.get(data_type, "STRING")
    return f"ARRAY<{base}>" if many else base


def tables() -> list[Table]:
    """All tables of the model, sorted by class name."""
    classes, properties = _read()
    by_domain: dict[str, list[_Property]] = {}
    for prop in properties.values():
        by_domain.setdefault(prop.domain, []).append(prop)
    out = []
    for name, entry in sorted(classes.items()):
        if "concrete" not in entry.stereotypes:
            continue
        chain = []
        current: str | None = name
        while current is not None and current in classes:
            chain.append(current)
            current = classes[current].parent
        columns = [
            Column(
                p.name, _sql_type(p, properties), p.multiplicity, tuple(sorted(p.profiles)), p.range
            )
            for ancestor in reversed(chain)
            for p in sorted(by_domain.get(ancestor, []), key=lambda p: p.name)
        ]
        out.append(Table(name, tuple(sorted(entry.profiles)), tuple(chain[1:]), columns))
    return out


def create_statement(table: Table, schema: str) -> str:
    """``CREATE TABLE`` for one class, with the published names as comments."""
    columns = ", ".join(f"`{c.name}` {c.sql_type} COMMENT '{c.cim_name}'" for c in table.columns)
    return (
        f"CREATE TABLE IF NOT EXISTS {schema}.{table.name.lower()} ({columns}) USING DELTA "
        f"COMMENT 'CGMES 3.0 class {table.name}'"
    )
