# Assembling a historical competition bundle

The harness treats `private/solution.csv` and `private/leaderboard.csv` as
read-only inputs assembled once per competition by a human. Steps:

1. **Data.** `kaggle competitions download -c <comp>` (check the competition's
   rules allow data download post-close), unzip `train.csv`, `test.csv`,
   `sample_submission.csv` into `competitions/<slug>/data/`.
2. **Private labels.** For Playground Series episodes the full solution is not
   published; reconstruct it from the source dataset where the episode page
   documents one, or use a competition where the organizer released the
   test labels. Save as `private/solution.csv` with columns
   `<id_column>,<target_column>` covering every test id.
3. **Final leaderboard.** Download the private leaderboard
   (`kaggle competitions leaderboard -c <comp> --download`) and save it as
   `private/leaderboard.csv` with at least a `score` column (one row per
   team, final private scores).
4. **Bundle.** Copy `bundle.yaml` from an existing comp, set `metric`,
   `metric_direction`, `target_column`, `id_column`, `cv_policy`, and the
   `leaderboard`/`private_labels` paths.
5. `medalist validate <slug>` must pass before anything else runs.

Note: `playground-s3e14/private/` is currently empty — grading for that comp
is blocked until steps 2–3 are done.
