"""Layering rules: the generic core never imports the domain layer, and runs without extras."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
OPTIONAL = (
    "shap",
    "lime",
    "dalex",
    "tensorflow",
    "keras",
    "xgboost",
    "lightgbm",
    "matplotlib",
    "skops",
    "powerfactory",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("path", sorted((SRC / "tranpy_core").rglob("*.py")), ids=lambda p: p.name)
def test_core_never_imports_domain_layer(path: Path) -> None:
    assert "tranpy" not in _imports(path)


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_optional_backends_are_not_imported_at_module_level(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert not top_level & set(OPTIONAL), f"{path.name} imports an optional backend at import time"


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=SRC.parent,
        check=False,
    )


def test_core_and_domain_work_without_optional_extras() -> None:
    blocked = repr(OPTIONAL)
    result = _run(
        f"""
        import importlib.abc, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split('.')[0] in {blocked}:
                    raise ImportError('blocked ' + name)
        sys.meta_path.insert(0, Block())
        import tranpy, tranpy_core
        from tranpy_core import ExperimentSpec, ModelConfig, run_experiment
        from tranpy_core.models import available_models
        from tranpy_core.errors import MissingDependencyError
        assert 'xgboost' not in available_models(installed_only=True)
        spec = ExperimentSpec(name='bare', models=(ModelConfig('logistic_regression'),))
        result = run_experiment(spec, tranpy.load_ieee_9bus())
        assert result.test_scores['logistic_regression'].accuracy > 0.9
        from tranpy_core.explain import explain_shap
        try:
            explain_shap(None, [[0.0]], background=[[0.0]])
        except MissingDependencyError as e:
            assert 'tranpy-ai[explain]' in str(e)
        else:
            raise SystemExit('expected MissingDependencyError')
        loaded = [m for m in sys.modules if m.split('.')[0] in {blocked}]
        assert not loaded, loaded
        print('ok')
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_importing_tranpy_does_not_load_heavy_modules() -> None:
    result = _run(
        """
        import sys, tranpy, tranpy_core, tranpy_core.explain, tranpy.simulation
        names = ('tensorflow', 'shap', 'lime', 'dalex', 'matplotlib', 'xgboost', 'lightgbm')
        heavy = [m for m in names if m in sys.modules]
        print(heavy)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
