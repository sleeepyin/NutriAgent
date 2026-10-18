#!/usr/bin/env python3
"""Download NutritionVerse-Real into data/nvreal. The directory is gitignored.

Kaggle serves a short-lived signed URL. This refreshes that URL and resumes
the zip until the archive is complete, then unpacks it.
"""
from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path


DATASET = "https://www.kaggle.com/api/v1/datasets/download/nutritionverse/nutritionverse-real"
ROOT = Path(__file__).resolve().parents[1] / "data" / "nvreal"
ZIP = ROOT / "nutritionverse-real.zip"
RAW = ROOT / "raw"


def signed_url() -> str:
    req = urllib.request.Request(DATASET, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.geturl()


def complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1_000_000_000:
        return False
    probe = subprocess.run(["unzip", "-t", str(path)], capture_output=True, text=True)
    return probe.returncode == 0


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    if complete(ZIP):
        print("zip already valid", ZIP, flush=True)
    else:
        for attempt in range(1, 21):
            url = signed_url()
            print(f"attempt {attempt} size={ZIP.stat().st_size if ZIP.exists() else 0}", flush=True)
            proc = subprocess.run(
                ["wget", "-c", "--timeout=60", "--read-timeout=60", "-O", str(ZIP), url],
            )
            if complete(ZIP):
                break
            if proc.returncode != 0:
                print(f"wget exit {proc.returncode}", flush=True)
        else:
            print("download did not finish a valid zip", flush=True)
            return 1
    subprocess.run(["unzip", "-n", "-q", str(ZIP), "-d", str(RAW)], check=True)
    print("unpacked", RAW, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
