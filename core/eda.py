"""Built-in EDA report: dtypes, missingness, cardinality, target distribution,
correlations, and a train/test drift check via adversarial validation AUC.

Usable as `python -m core.eda <slug>` (SPEC §9, Orient step).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from core.competition import Competition, load_bundle
from core.templates import render

ADVERSARIAL_SEED = 42


def _adversarial_auc(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> float:
    """AUC of a classifier told to separate train rows from test rows."""
    combined = pd.concat([train[feature_cols], test[feature_cols]], ignore_index=True)
    for col in combined.columns:
        if not pd.api.types.is_numeric_dtype(combined[col]):
            combined[col] = combined[col].astype("category").cat.codes
    labels = np.concatenate([np.zeros(len(train)), np.ones(len(test))])
    clf = HistGradientBoostingClassifier(max_iter=50, random_state=ADVERSARIAL_SEED)
    scores = cross_val_score(clf, combined, labels, cv=3, scoring="roc_auc")
    return float(scores.mean())


def write_eda_report(comp: Competition, out_dir: Path) -> Path:
    train = comp.load_train()
    test = comp.load_test()
    feature_cols = [c for c in train.columns if c not in (comp.target_column, comp.id_column)]

    columns: list[dict[str, Any]] = []
    for name in train.columns:
        columns.append(
            {
                "name": name,
                "dtype": str(train[name].dtype),
                "missing_train": 100.0 * float(train[name].isna().mean()),
                "missing_test": (
                    f"{100.0 * float(test[name].isna().mean()):.2f}" if name in test else "n/a"
                ),
                "cardinality": int(train[name].nunique()),
            }
        )

    y = train[comp.target_column]
    y_numeric = pd.api.types.is_numeric_dtype(y)
    target = {
        "mean": float(y.mean()) if y_numeric else float("nan"),
        "std": float(y.std()) if y_numeric else float("nan"),
        "min": float(y.min()) if y_numeric else float("nan"),
        "max": float(y.max()) if y_numeric else float("nan"),
        "nunique": int(y.nunique()),
        "looks_classification": y.nunique() < 20,
    }

    correlations: list[tuple[str, float]] = []
    if y_numeric:
        for name in feature_cols:
            if pd.api.types.is_numeric_dtype(train[name]) and train[name].std() > 0:
                r = float(np.corrcoef(train[name].fillna(train[name].mean()), y)[0, 1])
                correlations.append((name, r))
        correlations.sort(key=lambda item: abs(item[1]), reverse=True)

    auc = _adversarial_auc(train, test, feature_cols)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "eda.md"
    report_path.write_text(
        render(
            "eda.md.j2",
            comp=comp,
            n_train=len(train),
            n_test=len(test),
            columns=columns,
            target=target,
            correlations=correlations[:15],
            adversarial_auc=auc,
        )
    )
    return report_path


def main(argv: list[str]) -> None:
    slug = argv[0]
    root = Path(argv[1]) if len(argv) > 1 else Path.cwd()
    comp_dir = root / "competitions" / slug
    comp = load_bundle(comp_dir if comp_dir.is_dir() else Path(slug))
    path = write_eda_report(comp, root / "reports" / comp.slug)
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
