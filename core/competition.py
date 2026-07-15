"""Competition bundle loading & validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from core.scoring import METRIC_NAMES

CV_POLICIES = frozenset({"auto", "kfold", "stratified", "timeseries", "group"})
REQUIRED_FILES = ("train", "test", "sample_submission")


class BundleError(Exception):
    """Raised when a competition bundle fails validation."""


@dataclass(frozen=True)
class Competition:
    slug: str
    title: str
    metric: str
    metric_direction: str
    target_column: str
    id_column: str
    files: dict[str, Path]
    cv_policy: str
    cv_params: dict[str, Any] = field(default_factory=dict)
    time_column: str | None = None
    group_column: str | None = None
    leaderboard: Path | None = None
    private_labels: Path | None = None
    root: Path = Path(".")

    def load_train(self) -> pd.DataFrame:
        return pd.read_csv(self.files["train"])

    def load_test(self) -> pd.DataFrame:
        return pd.read_csv(self.files["test"])

    def load_sample_submission(self) -> pd.DataFrame:
        return pd.read_csv(self.files["sample_submission"])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def load_bundle(comp_dir: Path) -> Competition:
    """Load and validate bundle.yaml in comp_dir. Fail fast with actionable errors."""
    comp_dir = Path(comp_dir)
    bundle_path = comp_dir / "bundle.yaml"
    _require(bundle_path.is_file(), f"No bundle.yaml found at {bundle_path}")

    raw: dict[str, Any] = yaml.safe_load(bundle_path.read_text())
    _require(isinstance(raw, dict), f"bundle.yaml at {bundle_path} is not a mapping")

    for key in ("slug", "metric", "metric_direction", "target_column", "id_column", "files"):
        _require(key in raw, f"bundle.yaml missing required key '{key}'")

    metric = str(raw["metric"])
    _require(
        metric in METRIC_NAMES,
        f"Metric '{metric}' is not registered; known metrics: {sorted(METRIC_NAMES)}",
    )
    direction = str(raw["metric_direction"])
    _require(
        direction in ("minimize", "maximize"),
        f"metric_direction must be 'minimize' or 'maximize', got '{direction}'",
    )

    cv_policy = str(raw.get("cv_policy", "auto"))
    _require(cv_policy in CV_POLICIES, f"Unknown cv_policy '{cv_policy}'; one of {sorted(CV_POLICIES)}")
    time_column = raw.get("time_column") or None
    group_column = raw.get("group_column") or None
    _require(
        cv_policy != "timeseries" or time_column is not None,
        "cv_policy 'timeseries' requires time_column to be set",
    )
    _require(
        cv_policy != "group" or group_column is not None,
        "cv_policy 'group' requires group_column to be set",
    )

    files: dict[str, Path] = {}
    for name in REQUIRED_FILES:
        _require(name in raw["files"], f"files.{name} missing from bundle.yaml")
        path = comp_dir / str(raw["files"][name])
        _require(path.is_file(), f"files.{name} does not exist at {path}")
        files[name] = path

    target = str(raw["target_column"])
    id_col = str(raw["id_column"])

    train_head = pd.read_csv(files["train"], nrows=5)
    _require(target in train_head.columns, f"target_column '{target}' not in train columns")
    _require(id_col in train_head.columns, f"id_column '{id_col}' not in train columns")
    test_head = pd.read_csv(files["test"], nrows=5)
    _require(id_col in test_head.columns, f"id_column '{id_col}' not in test columns")

    try:
        sample_head = pd.read_csv(files["sample_submission"], nrows=5)
    except Exception as exc:
        raise BundleError(f"sample_submission failed to parse: {exc}") from exc
    _require(
        id_col in sample_head.columns and target in sample_head.columns,
        f"sample_submission must contain columns '{id_col}' and '{target}', "
        f"got {list(sample_head.columns)}",
    )

    def _opt_path(key: str) -> Path | None:
        value = raw.get(key) or None
        return comp_dir / str(value) if value is not None else None

    return Competition(
        slug=str(raw["slug"]),
        title=str(raw.get("title", raw["slug"])),
        metric=metric,
        metric_direction=direction,
        target_column=target,
        id_column=id_col,
        files=files,
        cv_policy=cv_policy,
        cv_params=dict(raw.get("cv_params") or {}),
        time_column=str(time_column) if time_column is not None else None,
        group_column=str(group_column) if group_column is not None else None,
        leaderboard=_opt_path("leaderboard"),
        private_labels=_opt_path("private_labels"),
        root=comp_dir,
    )
