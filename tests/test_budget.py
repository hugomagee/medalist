import pytest

from core.budget import Budget, BudgetExhausted


def test_defaults_match_spec() -> None:
    b = Budget()
    assert b.max_experiments == 25
    assert b.per_experiment_seconds == 900
    assert b.total_wall_seconds == 4 * 3600
    assert b.min_reserve_experiments == 2


def test_remaining_experiments() -> None:
    b = Budget(max_experiments=10, min_reserve_experiments=2)
    assert b.remaining_experiments(used=3) == 7


def test_queue_refused_when_only_reserve_left() -> None:
    b = Budget(max_experiments=10, min_reserve_experiments=2)
    b.check_can_queue(used=7)  # 8th of 10: fine (leaves 2 reserved)
    with pytest.raises(BudgetExhausted, match="reserve"):
        b.check_can_queue(used=8)


def test_reserved_slots_usable_for_final() -> None:
    b = Budget(max_experiments=10, min_reserve_experiments=2)
    b.check_can_queue(used=9, final=True)
    with pytest.raises(BudgetExhausted):
        b.check_can_queue(used=10, final=True)


def test_wall_clock_exhaustion() -> None:
    b = Budget(total_wall_seconds=100)
    b.check_wall(elapsed=99)
    with pytest.raises(BudgetExhausted, match="wall"):
        b.check_wall(elapsed=100)


def test_summary_mentions_remaining() -> None:
    b = Budget(max_experiments=10)
    text = b.summary(used=4, elapsed=60)
    assert "6" in text  # remaining experiments
    assert "budget" in text.lower()
