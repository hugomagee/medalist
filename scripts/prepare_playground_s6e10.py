"""One-off prep for playground-series-s6e10 (live competition, see DECISIONS.md).

Live competition, so NO holdout carve-out: the agent trains on the full
Kaggle train and predicts the real Kaggle test; grading is real submissions.

  raw/train.csv, raw/test.csv, raw/sample_submission.csv  untouched Kaggle files
  raw/original/data.csv                                   untouched public source
                                                          (arseniyshutko/binary-aviation-satisfaction-129k)
  data/train.csv          Kaggle train, target True/False -> 1/0
  data/test.csv           Kaggle test, unchanged
  data/sample_submission.csv  unchanged (id, satisfaction)
  data/original.csv       original rows, columns ordered like train, target -> 1/0,
                          plus a synthetic id column (orig_0..)

Idempotent: re-running overwrites data/ from raw/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COMP = Path(__file__).resolve().parents[1] / "competitions" / "playground-series-s6e10"
TARGET = "satisfaction"
ID = "id"


def _to01(s: pd.Series) -> pd.Series:
    mapped = s.astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    if mapped.isna().any():
        raise SystemExit(f"unexpected {TARGET} values: {s.unique()[:10]}")
    return mapped.astype("int64")


def main() -> None:
    raw, data = COMP / "raw", COMP / "data"
    data.mkdir(exist_ok=True)

    train = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")
    sample = pd.read_csv(raw / "sample_submission.csv")
    original = pd.read_csv(raw / "original" / "data.csv")

    train[TARGET] = _to01(train[TARGET])
    original[TARGET] = _to01(original[TARGET])

    feats = [c for c in train.columns if c not in (ID, TARGET)]
    missing = set(feats) ^ (set(original.columns) - {TARGET})
    if missing:
        raise SystemExit(f"original columns do not align with train: {missing}")
    original = original[feats + [TARGET]]
    original.insert(0, ID, [f"orig_{i}" for i in range(len(original))])

    train.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    sample.to_csv(data / "sample_submission.csv", index=False)
    original.to_csv(data / "original.csv", index=False)
    print(
        f"train={len(train)} (pos {train[TARGET].mean():.4f}) test={len(test)} "
        f"original={len(original)} (pos {original[TARGET].mean():.4f})"
    )


if __name__ == "__main__":
    main()
