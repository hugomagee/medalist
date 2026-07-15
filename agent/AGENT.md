# AGENT.md — operating manual for the competing agent

You are the scientist; the harness is the lab. It enforces honest CV, budget
limits, and a persistent memory so you can focus on hypotheses. Work the loop
below. Never fight the harness — if it refuses something, that refusal is
information.

## The loop

1. **Orient.** Read `competitions/<slug>/bundle.yaml` and
   `competitions/<slug>/MEMORY.md`. First session only: run
   `python -m core.eda <slug>` and read `reports/<slug>/eda.md`
   (dtypes, missingness, target, correlations, train/test drift AUC).
2. **Hypothesize.** State ONE falsifiable hypothesis (≥ 20 characters — the
   ledger rejects less). Check the graveyard in MEMORY.md first: families
   there have failed twice; do not retry them.
3. **Experiment.** Write `experiments/<slug>/<exp-id>/run.py` and
   `meta.yaml`, then queue it:

   ```
   medalist run <slug> --exp <exp-id>
   ```

   `meta.yaml`:

   ```yaml
   hypothesis: "target-encoding the shop column captures per-shop level shifts"
   approach: {family: gbm-fe, model_family: lightgbm, features_added: [shop_te]}
   parent_id: e0003   # experiment this one mutates, if any
   ```

   `run.py` contract:

   ```python
   def run(train_df, test_df, cv_splitter, metric):
       folds = cv_splitter.split(train_df)   # harness-fixed folds — use them
       ...
       return {
           "oof_predictions": oof,          # len == len(train_df)
           "test_predictions": test_pred,   # len == len(test_df)
           "fold_scores": [...],            # informational only
           "feature_importances": {...},    # optional
           "extra": {...},                  # optional
       }
   ```

4. **Analyze.** Read `experiments/<slug>/<exp-id>/result.json`. If the score
   improved, do error analysis (worst-predicted rows, residual patterns,
   importances.json) and record findings via the ledger notes.
5. **Iterate or ensemble.** When single-model gains flatten (< 0.2% relative
   improvement over 3 experiments), switch to ensembling the OOF predictions
   of the top-k *diverse* experiments (`strategies/ensembling.py`).
6. **Finalize.** `medalist submit <slug>` builds and validates the submission
   from the best experiment; `medalist grade <slug>` (historical comps)
   reports score, rank, percentile, and medal band in
   `reports/<slug>/RESULT.md`.

## Mandated discipline

- **Never** fit anything on the full train set before CV is finalized for
  that pipeline.
- Target encoding and imputation must be fit inside folds — use
  `strategies/features.py` (`target_encode_oof`, `impute_numeric_oof`),
  which are fold-safe by construction.
- The first three experiments are always:
  (a) naive baseline (`strategies.baselines.constant_baseline`),
  (b) LightGBM on raw features,
  (c) LightGBM + generic FE.
  These anchor the ledger.
- Determinism: the runner seeds `random`, `numpy`, and `PYTHONHASHSEED` to 42
  in your subprocess. Pass `random_state`/`seed` 42 to every model
  (`strategies.baselines.lgbm_default_params` does this for LightGBM).
- The `private/` directory is off-limits: `run.py` containing the string
  "private" is rejected before execution, the directory is unreadable while
  you run, and the harness — not you — computes every ledger score from your
  OOF predictions. Do not try.
- Budget is printed by every command. When only the reserve remains, spend it
  on the final ensemble (`medalist run ... --final`) and the submission.
