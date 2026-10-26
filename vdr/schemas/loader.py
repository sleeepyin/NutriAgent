"""Load frozen JSON Schemas and validate instances."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_DIR = _REPO_ROOT / "schemas"

VDR_OUTPUT_SCHEMA_PATH = _SCHEMA_DIR / "vdr_output.schema.json"
IR_STEP_SCHEMA_PATH = _SCHEMA_DIR / "ir_step.schema.json"
ACTIONS_PATH = _SCHEMA_DIR / "actions.json"


def load_schema(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_actions() -> dict[str, Any]:
    return load_schema(ACTIONS_PATH)


def validate_instance(instance: dict[str, Any], schema: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if instance is invalid."""
    Draft202012Validator(schema).validate(instance)
