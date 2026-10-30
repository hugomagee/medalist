"""Builds reports/playground-series-s6e10/{experiments.csv,progression.png,cv_vs_lb.csv}
from the ledger and the Kaggle submission list (kaggle CLI)."""
import csv
import io
import sqlite3
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SLUG = "playground-series-s6e10"
OUT = ROOT / "reports" / SLUG
OUT.mkdir(parents=True, exist_ok=True)

con = sqlite3.connect(ROOT / "ledger.db")
rows = con.execute(
    "select exp_id, status, cv_score_mean, hypothesis, coalesce(notes,'') from experiments "
    "where comp_slug=? order by exp_id", (SLUG,)).fetchall()
with open(OUT / "experiments.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["exp_id", "status", "cv_auc", "hypothesis", "notes"])
    w.writerows(rows)

done = [(r[0], r[2]) for r in rows if r[1] == "completed" and r[2] and r[2] > 0.9]
best, xs, ys, bx, by = 0.0, [], [], [], []
for i, (e, s) in enumerate(done):
    xs.append(int(e[1:])); ys.append(s)
    best = max(best, s); bx.append(int(e[1:])); by.append(best)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.scatter(xs, ys, s=18, color="#7a8ca5", label="experiment CV AUC")
ax.step(bx, by, where="post", color="#1f4e8c", lw=2, label="best so far")
ax.axhline(0.96209 + 0.0005, color="#b5651d", ls="--", lw=1, label="public plateau 0.9621 (+0.0005 CV offset)")
ax.set_xlabel("experiment id"); ax.set_ylabel("5-fold CV ROC AUC")
ax.set_title("Playground S6E10: CV AUC by experiment")
ax.set_ylim(0.957, 0.9630); ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "progression.png", dpi=130)

sub = subprocess.run(["kaggle", "competitions", "submissions", "-c", SLUG, "--csv"],
                     capture_output=True, text=True).stdout
lines = [l for l in sub.splitlines() if l and not l.startswith("Warning")]
cv = {r[0]: r[2] for r in rows}
with open(OUT / "cv_vs_lb.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["file", "description", "cv_auc", "public_lb", "date"])
    for r in csv.DictReader(io.StringIO("\n".join(lines))):
        exp = r["fileName"].replace(".csv", "")
        w.writerow([r["fileName"], r["description"], cv.get(exp, ""), r.get("publicScore", ""), r["date"]])
print("wrote", OUT)
