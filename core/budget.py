"""Budget manager (SPEC §10).

Hard caps on experiment count and wall clock. The final
`min_reserve_experiments` slots are reserved for the ensemble/submission
build and only usable with final=True.
"""

from __future__ import annotations

from dataclasses import dataclass


class BudgetExhausted(Exception):
    """Raised when an action would exceed a budget cap."""


@dataclass(frozen=True)
class Budget:
    max_experiments: int = 25
    per_experiment_seconds: float = 900.0
    total_wall_seconds: float = 4 * 3600.0
    min_reserve_experiments: int = 2

    def remaining_experiments(self, used: int) -> int:
        return max(0, self.max_experiments - used)

    def check_can_queue(self, used: int, final: bool = False) -> None:
        remaining = self.remaining_experiments(used)
        if remaining <= 0:
            raise BudgetExhausted(
                f"experiment cap reached ({used}/{self.max_experiments}); nothing may be queued"
            )
        if not final and remaining <= self.min_reserve_experiments:
            raise BudgetExhausted(
                f"only the reserve remains ({remaining} of {self.max_experiments} slots, "
                f"{self.min_reserve_experiments} reserved for the final ensemble/submission); "
                "queue with final=True to use it"
            )

    def check_wall(self, elapsed: float) -> None:
        if elapsed >= self.total_wall_seconds:
            raise BudgetExhausted(
                f"total wall-clock budget exhausted "
                f"({elapsed:.0f}s of {self.total_wall_seconds:.0f}s)"
            )

    def summary(self, used: int, elapsed: float) -> str:
        return (
            f"budget: {self.remaining_experiments(used)} of {self.max_experiments} "
            f"experiments remaining ({self.min_reserve_experiments} reserved), "
            f"{max(0.0, self.total_wall_seconds - elapsed):.0f}s of "
            f"{self.total_wall_seconds:.0f}s wall clock left"
        )
