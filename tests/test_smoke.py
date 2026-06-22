"""Fast, dependency-light smoke tests (no DB / no API key needed).

Run directly: `python tests/test_smoke.py`. Exits non-zero on failure so it can
gate CI without pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts import STYLES, SYSTEM_PROMPT_BUDGET, build_tutor_system  # noqa: E402
import mastery  # noqa: E402


def test_prompt_stays_under_budget():
    # Even a pathological struggle summary must not blow the char budget.
    huge = "weak: " + ", ".join(f"concept{i}" for i in range(100))
    for style in STYLES:
        s = build_tutor_system(style, "fast", 8, "low", huge)
        assert len(s) <= SYSTEM_PROMPT_BUDGET, (style, len(s))


def test_prompt_changes_by_style():
    visual = build_tutor_system("visual", "normal", 6, "med")
    steps = build_tutor_system("step_by_step", "normal", 6, "med")
    assert visual != steps


def test_mastery_rules():
    assert mastery.next_difficulty(mastery.update_mastery(0.5, False)) in (
        "easier", "same",
    )
    score = 0.5
    for _ in range(2):
        score = mastery.update_mastery(score, True)
    assert mastery.next_difficulty(score) == "harder"
    assert 0.0 <= mastery.update_mastery(0.0, False) <= 1.0


def test_struggle_summary_is_compact():
    attempts = [
        {"concept": c, "correct": False}
        for c in ["a", "a", "b", "b", "c", "d", "e"]
    ]
    summary = mastery.struggle_summary(attempts, top_n=3)
    assert summary.startswith("weak: ")
    assert summary.count(",") <= 2  # at most 3 tags


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
