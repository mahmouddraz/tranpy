"""Command-line interface: ``tranpy <command> ...`` (see ``tranpy --help``)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tranpy.datasets import list_datasets, load_snapshots, read_event_csvs
from tranpy.pretrained import train_pretrained
from tranpy.simulation import MockEngine, ScenarioSampler, generate_events
from tranpy_core.config import load_yaml
from tranpy_core.data import save_signals, sha256_file
from tranpy_core.errors import TranpyError
from tranpy_core.experiment import ExperimentSpec, run_experiment

__all__ = ["ExperimentConfig", "main"]


@dataclass(frozen=True)
class ExperimentConfig:
    """YAML file for ``tranpy experiment``: a snapshot dataset and an experiment spec."""

    dataset: str
    experiment: ExperimentSpec


def _experiment(args: argparse.Namespace) -> None:
    config = load_yaml(args.config, ExperimentConfig)
    result = run_experiment(config.experiment, load_snapshots(config.dataset))
    for name, scores in result.test_scores.items():
        print(
            f"{name:24s} accuracy={scores.accuracy:.4f} f1={scores.f1:.4f} "
            f"roc_auc={scores.roc_auc:.4f}"
        )
    if args.output:
        print(f"written {result.save_json(args.output)}")


def _datasets(_: argparse.Namespace) -> None:
    print(list_datasets().to_string(index=False))


def _convert_events(args: argparse.Namespace) -> None:
    path = save_signals(read_event_csvs(args.csv_dir, args.system), args.output)
    print(f"written {path} sha256={sha256_file(path)}")


def _simulate(args: argparse.Namespace) -> None:
    engine = MockEngine(args.system, simulation_time_s=args.duration)
    sampler = ScenarioSampler(
        n_lines=engine.n_lines,
        n_loads=engine.n_loads,
        simulation_time_s=args.duration,
        clearing_cycles=args.clearing_cycles,
        nominal_frequency_hz=engine.system.nominal_frequency_hz,
    )
    dataset = generate_events(engine, sampler.sample(args.events, seed=args.seed))
    path = save_signals(dataset, args.output)
    print(f"written {path}: {dataset}")


def _train_pretrained(args: argparse.Namespace) -> None:
    entries = train_pretrained(args.output, models=args.models or None)
    print(json.dumps({k: v["test_accuracy"] for k, v in entries.items()}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tranpy", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("datasets", help="list available datasets")
    p.set_defaults(func=_datasets)

    p = sub.add_parser("experiment", help="run an experiment described in a YAML file")
    p.add_argument("config", type=Path)
    p.add_argument("-o", "--output", type=Path, help="write the result as JSON")
    p.set_defaults(func=_experiment)

    p = sub.add_parser("convert-events", help="convert PowerFactory result CSVs to .npz")
    p.add_argument("csv_dir", type=Path)
    p.add_argument("system")
    p.add_argument("output", type=Path)
    p.set_defaults(func=_convert_events)

    p = sub.add_parser("simulate", help="generate an event dataset with the mock engine")
    p.add_argument("system")
    p.add_argument("output", type=Path)
    p.add_argument("--events", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clearing-cycles", type=float, default=12.0)
    p.add_argument("--duration", type=float, default=10.0, help="simulated time in s")
    p.set_defaults(func=_simulate)

    p = sub.add_parser("train-pretrained", help="train and save the pretrained pipelines")
    p.add_argument("output", type=Path)
    p.add_argument("--models", nargs="*", help="registry names (default: all installed)")
    p.set_defaults(func=_train_pretrained)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    try:
        args.func(args)
    except TranpyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
