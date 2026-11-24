"""Command line for Phase 1 training and the one-shot holdout."""

from __future__ import annotations

import argparse
import logging

from digi_nutri.jsonio import read_json
from digi_nutri.orchestrate import run_eval, run_train
from digi_nutri.protocol import HOLDOUT_METRICS, OUTPUTS, RESULTS
from digi_nutri.report import render_report


def main() -> None:
    parser = argparse.ArgumentParser(description="DiGI-Nutri Phase 1")
    commands = parser.add_subparsers(dest="cmd", required=True)
    train = commands.add_parser("train", help="train teachers and B1/B2a/B2b/B3, then evaluate once")
    train.add_argument("--cpu", action="store_true")
    train.add_argument("--epochs-teachers", type=int, default=None)
    train.add_argument("--epochs-students", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--workers", type=int, default=None)
    evaluate = commands.add_parser("eval", help="official holdout once; refuses if metrics already exist")
    evaluate.add_argument("--cpu", action="store_true")
    commands.add_parser("report", help="rewrite REPORT.md from the existing holdout file")
    args = parser.parse_args()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUTPUTS / "train.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    if args.cmd == "train":
        run_train(
            cpu=args.cpu,
            epochs_teachers=args.epochs_teachers,
            epochs_students=args.epochs_students,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        return
    if args.cmd == "eval":
        run_eval(cpu=args.cpu)
        return
    if not HOLDOUT_METRICS.is_file():
        raise SystemExit("no holdout metrics to report")
    metrics = read_json(HOLDOUT_METRICS)
    (RESULTS / "REPORT.md").write_text(render_report(metrics), encoding="utf-8")
    (RESULTS / "verdict.txt").write_text(metrics["verdict"] + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
