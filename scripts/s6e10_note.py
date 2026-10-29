"""usage: uv run python scripts/s6e10_note.py <exp-id> "<note>"  (ledger notes API)"""
import sys
from pathlib import Path

from core.ledger import Ledger

Ledger(Path(__file__).resolve().parents[1] / "ledger.db").set_notes(sys.argv[1], sys.argv[2])
print("noted", sys.argv[1])
