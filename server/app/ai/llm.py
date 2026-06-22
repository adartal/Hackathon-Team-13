"""Thin model wrapper. The provider/model lives here ONLY.

MVP uses Google Gemini Flash (free tier via Google AI Studio). To move to a
paid model later (e.g. Claude Sonnet 4.6), swap the two functions below — no
other file imports the provider SDK.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types

from app.ai.prompts import (
    ANALYSIS_PROMPT,
    CONTEXT_SUMMARY_PROMPT,
    PLAN_PROMPT,
    AnalysisResult,
)
from app.schemas.tutor import TutoringPlan

# Free-tier, vision-capable, good at math. Confirm the current free model id
# at https://aistudio.google.com/ — kept here so it changes in one place.
MODEL = "gemini-3.1-flash-lite"

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=api_key)
    return _client


def analyze(image_bytes: bytes, mime_type: str = "image/jpeg") -> AnalysisResult:
    """Call A — structured grading of the photo (drives logging + mastery)."""
    resp = _get_client().models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ANALYSIS_PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AnalysisResult,
        ),
    )
    # The SDK parses response_schema into .parsed for us.
    parsed = resp.parsed
    if isinstance(parsed, AnalysisResult):
        return parsed
    return AnalysisResult.model_validate(parsed)


def plan(system: str, context: str) -> TutoringPlan:
    """Call B — structured private plan the tutor commits to before replying.

    Forcing a small schema-checked plan out of the weak free model yields far
    more deliberate pedagogy than a single unstructured reply call.
    """
    resp = _get_client().models.generate_content(
        model=MODEL,
        contents=[f"{PLAN_PROMPT}\n\n{context}"],
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=TutoringPlan,
            max_output_tokens=400,
        ),
    )
    parsed = resp.parsed
    if isinstance(parsed, TutoringPlan):
        return parsed
    return TutoringPlan.model_validate(parsed)


def tutor(system: str, context: str) -> str:
    """Call C — the human-facing tutoring reply (bounded system prompt)."""
    resp = _get_client().models.generate_content(
        model=MODEL,
        contents=[context],
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=600,
        ),
    )
    return (resp.text or "").strip()


def generate_practice(concept_he: str, difficulty: str, grade: int = 8) -> str:
    """On-demand: invent one fresh practice problem for a concept + difficulty.

    Grade-aligned so the problem stays at the student's middle-school level.
    Deliberately OFF the per-turn path — only the practice endpoints call this
    when explicitly asked, so per-turn LLM cost stays unchanged.
    """
    resp = _get_client().models.generate_content(
        model=MODEL,
        contents=[
            "You are a math teacher writing a practice problem for an Israeli middle-school student. "
            f"Topic: '{concept_he}', grade: {grade}, "
            f"difficulty: {difficulty} (easier/same/harder than typical for the grade).\n\n"
            "Write ONE short, self-contained practice problem in Hebrew.\n\n"
            "FORMATTING RULES — follow exactly:\n"
            "- Write the prose in Hebrew (right-to-left text).\n"
            "- For ALL mathematical expressions use LaTeX wrapped in single dollar signs: $expression$\n"
            "  Examples: $x^2$, $\\frac{1}{2}$, $\\sqrt{16}$, $\\parallel$, $\\perp$, $3 \\cdot x$\n"
            "- Never use Unicode math characters like ², ½, √, ‖ — always use LaTeX $...$\n"
            "- Do NOT add a title, solution, or explanation — only the problem statement."
        ],
        config=types.GenerateContentConfig(max_output_tokens=150),
    )
    return (resp.text or "").strip()


def generate_from_prompt(teacher_prompt: str) -> str:
    """Generate a Hebrew math question from a free-form teacher prompt.

    Unlike generate_practice (which needs a concept + difficulty), this accepts
    any descriptive text the teacher types (e.g. "easy fractions", "word problem
    with percentages for grade 6") and returns a single problem statement.
    """
    resp = _get_client().models.generate_content(
        model=MODEL,
        contents=[
            "You are a math teacher writing a problem for an Israeli middle-school student. "
            f"Teacher's request: '{teacher_prompt}'. "
            "Write ONE clear, self-contained math problem statement in Hebrew that matches the request.\n\n"
            "FORMATTING RULES — follow exactly:\n"
            "- Write the prose in Hebrew (right-to-left text).\n"
            "- For ALL mathematical expressions use LaTeX wrapped in single dollar signs: $expression$\n"
            "  Examples: $x^2$, $\\frac{1}{2}$, $\\sqrt{16}$, $\\parallel$, $\\perp$, $\\cdot$\n"
            "- Never use Unicode math characters like ², ½, √, ‖ — always use LaTeX $...$\n"
            "- Do NOT add a title, solution, or explanation — only the problem statement."
        ],
        config=types.GenerateContentConfig(max_output_tokens=200),
    )
    return (resp.text or "").strip()


def summarize(prev_summary: str, latest_exchange: str) -> str:
    """Compact the running conversation memory after each turn (context.json)."""
    resp = _get_client().models.generate_content(
        model=MODEL,
        contents=[
            f"{CONTEXT_SUMMARY_PROMPT}\n\n"
            f"Previous summary:\n{prev_summary or '(none yet)'}\n\n"
            f"Latest exchange:\n{latest_exchange}"
        ],
        config=types.GenerateContentConfig(max_output_tokens=200),
    )
    return (resp.text or "").strip()
