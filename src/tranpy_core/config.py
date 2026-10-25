"""Typed configuration: frozen dataclasses built from mappings or YAML files.

``from_mapping`` converts plain data (e.g. parsed YAML) into a frozen dataclass, rejects
unknown keys and reports *all* problems at once. Dataclasses validate their own values in
``__post_init__``; such errors are collected too.
"""

from __future__ import annotations

import dataclasses
import enum
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args, get_origin, get_type_hints

import yaml

from tranpy_core.errors import ConfigurationError

__all__ = ["from_mapping", "load_yaml", "to_mapping"]

T = TypeVar("T")

_MISSING = object()


def load_yaml(path: str | Path, cls: type[T]) -> T:
    """Read a YAML file and convert it with :func:`from_mapping`."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read configuration file {path}: {exc}") from exc
    if data is None:
        data = {}
    return from_mapping(cls, data, source=str(path))


def from_mapping(cls: type[T], data: Any, *, source: str = "configuration") -> T:
    """Build ``cls`` (a dataclass) from ``data``; raise one error listing every problem."""
    problems: list[str] = []
    result = _convert(cls, data, "", problems)
    if problems:
        raise ConfigurationError(f"invalid {source}:", tuple(problems))
    return typing.cast(T, result)


def to_mapping(obj: Any) -> Any:
    """Convert a dataclass tree to plain YAML/JSON-friendly data."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_mapping(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, Mapping):
        return {str(k): to_mapping(v) for k, v in obj.items()}
    if isinstance(obj, tuple | list):
        return [to_mapping(v) for v in obj]
    return obj


def _where(path: str) -> str:
    return path or "<root>"


def _convert(tp: Any, value: Any, path: str, problems: list[str]) -> Any:
    origin = get_origin(tp)
    if tp is Any:
        return value
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        return _convert_dataclass(tp, value, path, problems)
    if origin in (typing.Union, types.UnionType):
        options = get_args(tp)
        if value is None and type(None) in options:
            return None
        for option in (o for o in options if o is not type(None)):
            trial: list[str] = []
            converted = _convert(option, value, path, trial)
            if not trial:
                return converted
        problems.append(f"{_where(path)}: {value!r} does not match {_type_name(tp)}")
        return None
    if origin is Literal:
        if value not in get_args(tp):
            allowed = ", ".join(repr(a) for a in get_args(tp))
            problems.append(f"{_where(path)}: {value!r} is not one of {allowed}")
        return value
    if origin is tuple:
        return _convert_tuple(tp, value, path, problems)
    if origin in (dict, Mapping) or tp in (dict, Mapping):
        return _convert_mapping(tp, value, path, problems)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        try:
            return tp(value)
        except ValueError:
            allowed = ", ".join(repr(m.value) for m in tp)
            problems.append(f"{_where(path)}: {value!r} is not one of {allowed}")
            return None
    if tp is float:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif tp is type(None):
        if value is None:
            return None
    elif isinstance(tp, type) and isinstance(value, tp):
        return value
    problems.append(f"{_where(path)}: expected {_type_name(tp)}, got {value!r}")
    return None


def _convert_dataclass(cls: type[Any], value: Any, path: str, problems: list[str]) -> Any:
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        problems.append(f"{_where(path)}: expected a mapping for {cls.__name__}, got {value!r}")
        return None
    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    for key in value:
        if key not in fields:
            known = ", ".join(sorted(fields))
            problems.append(f"{path + '.' if path else ''}{key}: unknown key (known: {known})")
    kwargs: dict[str, Any] = {}
    n_before = len(problems)
    for name, field in fields.items():
        sub = f"{path}.{name}" if path else name
        if name in value:
            kwargs[name] = _convert(hints[name], value[name], sub, problems)
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            problems.append(f"{sub}: required key is missing")
    if len(problems) > n_before:
        return None
    try:
        return cls(**kwargs)
    except (ConfigurationError, ValueError, TypeError) as exc:
        details = getattr(exc, "problems", ()) or (str(exc),)
        problems.extend(f"{_where(path)}: {d}" for d in details)
        return None


def _convert_tuple(tp: Any, value: Any, path: str, problems: list[str]) -> Any:
    if not isinstance(value, list | tuple):
        problems.append(f"{_where(path)}: expected a list, got {value!r}")
        return None
    args = get_args(tp)
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(_convert(args[0], v, f"{path}[{i}]", problems) for i, v in enumerate(value))
    if len(args) != len(value):
        problems.append(f"{_where(path)}: expected {len(args)} items, got {len(value)}")
        return None
    return tuple(
        _convert(a, v, f"{path}[{i}]", problems)
        for i, (a, v) in enumerate(zip(args, value, strict=True))
    )


def _convert_mapping(tp: Any, value: Any, path: str, problems: list[str]) -> Any:
    if not isinstance(value, Mapping):
        problems.append(f"{_where(path)}: expected a mapping, got {value!r}")
        return None
    args = get_args(tp) or (str, Any)
    key_tp, val_tp = args
    return types.MappingProxyType(
        {
            _convert(key_tp, k, f"{path}<key>", problems): _convert(
                val_tp, v, f"{path}.{k}", problems
            )
            for k, v in value.items()
        }
    )


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp).replace("typing.", "")
