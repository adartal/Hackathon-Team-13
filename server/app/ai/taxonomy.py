"""Canonical middle-school math concept taxonomy (pure, no I/O).

The grader (Call A) emits a free-form ``concept`` string; the free model is not
trusted to stay on a fixed enum. So we keep a small *canonical* curriculum here
and run every raw tag through :func:`normalize_concept` before it touches
mastery. Canonical ids are the source of truth: they let per-concept mastery
aggregate cleanly across turns and power the (in-progress) teacher portal, and
the ``prerequisites`` graph lets the proactive next-step recommender redirect a
struggling student to the missing foundation.

Everything here is plain data + pure functions so it unit-tests without S3/LLM.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

# Two normalized problem signatures at or above this ratio are treated as the
# same problem (drives the within-problem hint ladder). Tuned to absorb
# transcription noise ("2/3 + 1/4" vs "2/3+1/4") without merging genuinely
# different problems.
SAME_PROBLEM_RATIO = 0.85


@dataclass(frozen=True)
class Concept:
    """One canonical curriculum concept."""

    id: str  # snake_case ASCII canonical key (the mastery dict key)
    he: str  # Hebrew display name (what a student/teacher reads)
    grade_band: tuple[int, int]  # inclusive (min_grade, max_grade)
    prerequisites: tuple[str, ...]  # canonical ids that should come first


# The curriculum. Small but real; ordered roughly foundation -> advanced so the
# prerequisite edges read naturally. "other" is a real entry so nothing has to
# special-case an unknown tag.
_CONCEPT_LIST: tuple[Concept, ...] = (
    Concept("whole_number_arithmetic", "חשבון מספרים שלמים", (1, 6), ()),
    Concept("integers", "מספרים שלמים ושליליים", (6, 8), ("whole_number_arithmetic",)),
    Concept("order_of_operations", "סדר פעולות חשבון", (5, 8), ("whole_number_arithmetic",)),
    Concept("fractions", "שברים", (4, 8), ("whole_number_arithmetic",)),
    Concept("decimals", "שברים עשרוניים", (4, 8), ("fractions",)),
    Concept("percentages", "אחוזים", (6, 9), ("fractions", "decimals")),
    Concept("ratios_proportions", "יחס ופרופורציה", (6, 9), ("fractions",)),
    Concept("exponents", "חזקות", (7, 9), ("whole_number_arithmetic",)),
    Concept("algebraic_expressions", "ביטויים אלגבריים", (7, 9), ("integers", "order_of_operations")),
    Concept("linear_equations", "משוואות ממעלה ראשונה", (7, 9), ("algebraic_expressions", "integers")),
    Concept("inequalities", "אי-שוויונות", (7, 9), ("linear_equations",)),
    Concept("coordinate_geometry", "מערכת צירים", (7, 9), ("integers",)),
    Concept("area_perimeter", "שטח והיקף", (4, 8), ("whole_number_arithmetic",)),
    Concept("volume", "נפח", (6, 9), ("area_perimeter",)),
    Concept("angles", "זוויות", (5, 8), ()),
    Concept("word_problems", "בעיות מילוליות", (4, 9), ("whole_number_arithmetic",)),
    Concept("statistics", "סטטיסטיקה", (6, 9), ("fractions",)),
    Concept("other", "נושא אחר", (1, 12), ()),
)

CONCEPTS: dict[str, Concept] = {c.id: c for c in _CONCEPT_LIST}

# Lowercased raw grader tags (English or Hebrew) -> canonical id. Only the
# common variants the model actually emits; the token fallback in
# normalize_concept() catches the long tail.
_ALIASES: dict[str, str] = {
    "fraction": "fractions",
    "fraction addition": "fractions",
    "adding fractions": "fractions",
    "שבר": "fractions",
    "שברים": "fractions",
    "decimal": "decimals",
    "decimal numbers": "decimals",
    "percent": "percentages",
    "percentage": "percentages",
    "אחוז": "percentages",
    "ratio": "ratios_proportions",
    "ratios": "ratios_proportions",
    "proportion": "ratios_proportions",
    "proportions": "ratios_proportions",
    "negative numbers": "integers",
    "integer": "integers",
    "signed numbers": "integers",
    "sign": "integers",
    "exponent": "exponents",
    "powers": "exponents",
    "power": "exponents",
    "order of operations": "order_of_operations",
    "pemdas": "order_of_operations",
    "expression": "algebraic_expressions",
    "expressions": "algebraic_expressions",
    "algebra": "algebraic_expressions",
    "algebraic expression": "algebraic_expressions",
    "equation": "linear_equations",
    "equations": "linear_equations",
    "linear equation": "linear_equations",
    "solving equations": "linear_equations",
    "inequality": "inequalities",
    "coordinate": "coordinate_geometry",
    "coordinates": "coordinate_geometry",
    "graph": "coordinate_geometry",
    "area": "area_perimeter",
    "perimeter": "area_perimeter",
    "geometry": "area_perimeter",
    "volume": "volume",
    "angle": "angles",
    "angles": "angles",
    "word problem": "word_problems",
    "statistics": "statistics",
    "average": "statistics",
    "mean": "statistics",
    "arithmetic": "whole_number_arithmetic",
    "multiplication": "whole_number_arithmetic",
    "division": "whole_number_arithmetic",
    "addition": "whole_number_arithmetic",
    "subtraction": "whole_number_arithmetic",
}


def normalize_concept(raw: str) -> str:
    """Map a free-form grader tag to a canonical concept id.

    Resolution order: exact canonical id -> alias table -> token/substring
    overlap with a canonical id or alias -> ``"other"``. Never raises.
    """
    key = (raw or "").strip().lower()
    if not key:
        return "other"
    if key in CONCEPTS:
        return key
    if key in _ALIASES:
        return _ALIASES[key]

    # Token fallback: if any whitespace-delimited token is itself a known alias
    # or canonical id, use it (handles "fraction word problem" -> fractions).
    tokens = re.split(r"[\s/,_-]+", key)
    for tok in tokens:
        if tok in CONCEPTS:
            return tok
        if tok in _ALIASES:
            return _ALIASES[tok]
    return "other"


def candidate_ids() -> list[str]:
    """Canonical ids for grounding the analysis prompt (excludes ``other``)."""
    return [cid for cid in CONCEPTS if cid != "other"]


def catalog(grade: int | None = None) -> list[dict[str, object]]:
    """Pickable concepts (id + Hebrew name) for the practice subject picker.

    Excludes the ``other`` catch-all. When ``grade`` is given, concepts whose band
    covers that grade are flagged ``in_grade`` (and listed first) so the UI can
    surface grade-appropriate subjects without hiding the rest.
    """
    out: list[dict[str, object]] = []
    for c in _CONCEPT_LIST:
        if c.id == "other":
            continue
        in_grade = grade is None or c.grade_band[0] <= grade <= c.grade_band[1]
        out.append({"concept": c.id, "he_name": c.he, "in_grade": in_grade})
    out.sort(key=lambda d: (not d["in_grade"],))  # grade-appropriate first, stable otherwise
    return out


def display_name(cid: str) -> str:
    """Hebrew display name for a canonical id (falls back to the id)."""
    concept = CONCEPTS.get(cid)
    return concept.he if concept else cid


def prerequisites_of(cid: str) -> tuple[str, ...]:
    """Direct prerequisite ids for a concept ("" / unknown -> empty)."""
    concept = CONCEPTS.get(cid)
    return concept.prerequisites if concept else ()


# --- Problem-signature matching (within-problem hint ladder) -----------------

# Hebrew instruction verbs to strip so "חשב/י 2/3 + 1/4" and "2/3 + 1/4" match.
_INSTRUCTION_WORDS = (
    "חשב",
    "חשבי",
    "פתר",
    "פתרי",
    "מצא",
    "מצאי",
    "calculate",
    "solve",
    "find",
    "compute",
)
_INSTRUCTION_RE = re.compile("|".join(re.escape(w) for w in _INSTRUCTION_WORDS))
# Keep only math-bearing characters; drop everything else (words, punctuation).
_KEEP_RE = re.compile(r"[^0-9+\-*/=().xyπ%]+")


def problem_signature(raw: str) -> str:
    """Reduce a problem statement to a stable math-only signature.

    Strips Hebrew niqqud and instruction verbs, keeps only digits/operators/
    variables, and collapses whitespace, so cosmetic transcription differences
    don't reset the hint ladder. Returns "" for an empty/unclear problem.
    """
    s = (raw or "").strip().lower()
    if not s or s == "(unclear)":
        return ""
    # Drop combining marks (Hebrew niqqud / accents).
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = _INSTRUCTION_RE.sub(" ", s)
    s = _KEEP_RE.sub("", s)
    return s


def same_problem(sig_a: str, sig_b: str) -> bool:
    """True when two signatures refer to the same problem.

    Empty signatures never match (a fresh/unclear problem is always "new").
    """
    if not sig_a or not sig_b:
        return False
    if sig_a == sig_b:
        return True
    return SequenceMatcher(None, sig_a, sig_b).ratio() >= SAME_PROBLEM_RATIO
