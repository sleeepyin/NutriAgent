#!/usr/bin/env python3
from __future__ import annotations

import argparse

from .common import add_common_args, out_dir, save_status
from agent_extension.src.build_group_manifest import build_manifests
from agent_extension.src.paths import Paths


def main(argv=None) -> int:
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args(argv)
    paths = Paths.resolve()
    report = build_manifests(paths)
    d = out_dir("e00_manifest", args)
    save_status(d, bool(report.get("ok", True)), report=report)
    return 0 if report.get("ok", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
