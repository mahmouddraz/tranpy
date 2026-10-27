from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest

from tranpy_core.config import from_mapping, load_yaml, to_mapping
from tranpy_core.errors import ConfigurationError


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


@dataclass(frozen=True)
class Inner:
    rate: float
    kind: Literal["a", "b"] = "a"

    def __post_init__(self) -> None:
        if self.rate < 0:
            raise ConfigurationError("bad inner", ("rate must be >= 0",))


@dataclass(frozen=True)
class Outer:
    name: str
    inner: Inner
    sizes: tuple[int, ...] = ()
    pair: tuple[int, str] = (0, "x")
    color: Color = Color.RED
    extra: dict[str, Any] = field(default_factory=dict)
    maybe: int | None = None


def test_from_mapping_builds_nested_and_converts_types() -> None:
    cfg = from_mapping(
        Outer,
        {
            "name": "n",
            "inner": {"rate": 1, "kind": "b"},
            "sizes": [1, 2],
            "pair": [3, "y"],
            "color": "blue",
            "extra": {"k": [1]},
            "maybe": None,
        },
    )
    assert cfg.inner == Inner(rate=1.0, kind="b")
    assert isinstance(cfg.inner.rate, float)
    assert cfg.sizes == (1, 2)
    assert cfg.pair == (3, "y")
    assert cfg.color is Color.BLUE
    assert dict(cfg.extra) == {"k": [1]}


def test_all_problems_reported_at_once() -> None:
    with pytest.raises(ConfigurationError) as info:
        from_mapping(
            Outer,
            {
                "nme": "typo",
                "inner": {"rate": "fast", "kind": "c"},
                "sizes": [1, "two"],
                "pair": [1],
                "color": "green",
                "maybe": True,
            },
        )
    problems = info.value.problems
    text = "\n".join(problems)
    assert "nme: unknown key" in text
    assert "name: required key is missing" in text
    assert "inner.rate" in text
    assert "inner.kind" in text
    assert "sizes[1]" in text
    assert "pair: expected 2 items" in text
    assert "color" in text
    assert "maybe" in text
    assert len(problems) >= 8


def test_post_init_errors_are_collected() -> None:
    with pytest.raises(ConfigurationError, match="rate must be >= 0"):
        from_mapping(Outer, {"name": "n", "inner": {"rate": -1}})


def test_wrong_container_types() -> None:
    with pytest.raises(ConfigurationError) as info:
        from_mapping(Outer, {"name": "n", "inner": 3, "sizes": "abc", "extra": [1]})
    text = "\n".join(info.value.problems)
    assert "expected a mapping for Inner" in text
    assert "sizes: expected a list" in text
    assert "extra: expected a mapping" in text


def test_root_must_be_mapping() -> None:
    with pytest.raises(ConfigurationError, match="<root>"):
        from_mapping(Outer, [1, 2])


def test_existing_instance_passes_through() -> None:
    inner = Inner(rate=2.0)
    assert from_mapping(Outer, {"name": "n", "inner": inner}).inner is inner


def test_yaml_round_trip(tmp_path: Path) -> None:
    import yaml

    cfg = Outer(name="n", inner=Inner(rate=0.5), sizes=(1,), color=Color.BLUE)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(to_mapping(cfg)))
    assert load_yaml(path, Outer) == cfg


def test_load_yaml_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read"):
        load_yaml(tmp_path / "missing.yaml", Outer)
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(ConfigurationError, match="required key is missing"):
        load_yaml(empty, Outer)


def test_configuration_error_message_lists_problems() -> None:
    err = ConfigurationError("bad", ("a", "b"))
    assert str(err) == "bad\n  - a\n  - b"
    assert isinstance(err, ValueError)
