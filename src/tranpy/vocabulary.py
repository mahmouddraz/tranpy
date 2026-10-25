"""Power-system vocabulary: stability classes, network elements and recorded variables.

Channel names follow the PowerFactory result export, ``"<element>|<variable> in <unit>"``,
for example ``"G 01|c:firot in deg"`` (rotor angle of generator G 01 in degrees).

Generic <-> domain mapping used throughout the package:

=================  ==============================================
generic (core)     power system (this package)
=================  ==============================================
sample             simulated fault event
signal / channel   recorded variable of one network element
label 0 / 1        :attr:`Stability.STABLE` / :attr:`Stability.UNSTABLE`
annotation         event time, e.g. ``fault_start_s``
=================  ==============================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum, StrEnum

from tranpy_core.errors import DataError

__all__ = [
    "LABEL_NAMES",
    "ChannelName",
    "ElementKind",
    "Stability",
    "Variable",
    "element_kind",
]


class Stability(IntEnum):
    """Transient stability class of a fault event (the classification label)."""

    STABLE = 0
    UNSTABLE = 1


LABEL_NAMES: tuple[str, ...] = tuple(s.name.lower() for s in Stability)


class ElementKind(StrEnum):
    BUS = "bus"
    GENERATOR = "generator"
    LINE = "line"
    LOAD = "load"
    TRANSFORMER = "transformer"
    OTHER = "other"


class Variable(StrEnum):
    """PowerFactory result variables used by this package (name and unit as exported)."""

    BUS_VOLTAGE_PU = "m:u in p.u."
    BUS_ANGLE_DEG = "m:phiu in deg"
    BUS_FREQUENCY_HZ = "m:fehz in Hz"
    ROTOR_ANGLE_DEG = "c:firot in deg"
    """Rotor angle relative to the reference machine."""
    ROTOR_SPEED_PU = "s:xspeed in p.u."
    ROTOR_FREQUENCY_DEVIATION_HZ = "c:dorhz in Hz"
    ACTIVE_POWER_MW = "s:P1 in MW"
    REACTIVE_POWER_MVAR = "s:Q1 in Mvar"
    OUT_OF_STEP = "s:outofstep"
    """1 once the machine has lost synchronism (PowerFactory pole-slip flag)."""
    TIME_S = "b:tnow in s"


_GENERATOR = re.compile(r"^G\s*\d+$")
_TRANSFORMER = re.compile(r"^(T|Trf)\s*\d+")


def element_kind(element: str) -> ElementKind:
    """Classify an element name as exported by PowerFactory."""
    name = element.strip()
    if name.startswith("Bus"):
        return ElementKind.BUS
    if _GENERATOR.match(name):
        return ElementKind.GENERATOR
    if name.startswith("Line"):
        return ElementKind.LINE
    if name.startswith("Load"):
        return ElementKind.LOAD
    if _TRANSFORMER.match(name):
        return ElementKind.TRANSFORMER
    return ElementKind.OTHER


@dataclass(frozen=True)
class ChannelName:
    """A parsed channel name ``"<element>|<variable>"``."""

    element: str
    variable: str

    @classmethod
    def parse(cls, channel: str) -> ChannelName:
        element, sep, variable = channel.partition("|")
        if not sep or not element or not variable:
            raise DataError(f"channel name {channel!r} is not of the form 'element|variable'")
        return cls(element=element, variable=variable)

    @property
    def kind(self) -> ElementKind:
        return element_kind(self.element)

    def __str__(self) -> str:
        return f"{self.element}|{self.variable}"
