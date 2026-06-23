"""End-to-end test of the full Caliber pipeline against local phi3.

Runs the same DIRECT-vs-CoT comparison from earlier real_eval_* scripts, but
this time chains all three layers:

  1. caliber.compare()         — is the difference statistically real?
  2. caliber.explain()         — where did the gain come from? which examples drove it?
  3. caliber.judge_hypothesis() — what's the pattern? (phi3 reads the examples)

Output mirrors the previous test scripts: per-problem table, then verdict block,
then stratified analysis, driving examples, and finally the AI-generated
hypothesis for WHY.

Run:
    python -X utf8 real_eval_full_pipeline.py
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass

import numpy as np
import requests

import caliber

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3"

# 10 mixed problems across 4 categories (CoT should help all of them).
PROBLEMS: list[tuple[str, int, str]] = [
    ("Compute: 8 + 4 * 3 - 2",                                              18, "PEMDAS"),
    ("Compute: (12 + 6) * 4 - 10",                                          62, "PEMDAS"),
    ("Compute: 100 - 4^2 * 3",                                              52, "PEMDAS"),
    ("If 3x + 7 = 25, what is x?",                                           6, "algebra"),
    ("If 5x - 12 = 33, what is x?",                                          9, "algebra"),
    ("If 2(x + 4) = 26, what is x?",                                         9, "algebra"),
    ("The sum of three consecutive integers is 84. What is the smallest?",  27, "word"),
    ("Two numbers sum to 20 and differ by 4. What is the larger?",          12, "word"),
    ("What is 35 percent of 80?",                                           28, "percent"),
    ("If 45 percent of x equals 90, what is x?",                           200, "percent"),
]


PROMPT_DIRECT = "{q}\n\nReply with only the final number, no other text."
PROMPT_COT = (
    "{q}\n\nThink through this step by step. After your reasoning, write the "
    "final answer on its own line in the exact format: ANSWER = <number>"
)


def ask_phi3(prompt: str) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 600},
        },
        timeout=180,
    )
    r.raise_for_status()
    return str(r.json()["response"])


def extract_number(text: str, prefer_after_label: bool = False) -> int | None:
    if prefer_after_label:
        m = re.search(r"answer\s*[:=]\s*(-?\d[\d,]*)", text, re.IGNORECASE)
        if m:
            return int(m.group(1).replace(",", ""))
    nums = re.findall(r"-?\d[\d,]*", text)
    if not nums:
        return None
    try:
        return int(nums[-1].replace(",", ""))
    except ValueError:
        return None


@dataclass
class ArmResult:
    scores: np.ndarray
    responses: list[str]
    guesses: list[int | None]


def run_arm(template: str, label: str, is_cot: bool) -> ArmResult:
    print(f"\n--- {label} ---")
    scores = np.zeros(len(PROBLEMS), dtype=np.float64)
    responses: list[str] = []
    guesses: list[int | None] = []
    t0 = time.time()
    for i, (q, correct, _cat) in enumerate(PROBLEMS):
        response = ask_phi3(template.format(q=q))
        guess = extract_number(response, prefer_after_label=is_cot)
        score = 1.0 if guess == correct else 0.0
        scores[i] = score
        responses.append(response)
        guesses.append(guess)
        status = "OK " if score else "BAD"
        q_short = q if len(q) <= 50 else q[:47] + "..."
        guess_str = str(guess) if guess is not None else "?"
        print(f"  [{i + 1:2d}/{len(PROBLEMS)}] {status}  correct={correct:4d}  phi3 said={guess_str:>6}  | {q_short}")
    print(f"  accuracy: {scores.mean():.0%}  elapsed: {time.time() - t0:.1f}s")
    return ArmResult(scores, responses, guesses)


def main() -> int:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: Ollama not reachable ({e})", file=sys.stderr)
        return 1

    print(f"Caliber full-pipeline test — local {MODEL} via Ollama")
    print(f"Task: {len(PROBLEMS)} math problems "
          f"({len({p[2] for p in PROBLEMS})} categories)")
    print(f"Comparing: direct-answer vs chain-of-thought\n"
          "Pipeline:  compare() -> explain() -> judge_hypothesis()")

    direct = run_arm(PROMPT_DIRECT, "OLD: direct answer", is_cot=False)
    cot = run_arm(PROMPT_COT, "NEW: chain-of-thought", is_cot=True)

    # ========================================================================
    # Layer 1: compare() — the statistical verdict
    # ========================================================================
    print()
    print("=" * 72)
    print("LAYER 1 — caliber.compare()  (is the difference real?)")
    print("=" * 72)
    verdict = caliber.compare(
        direct.scores, cot.scores, metric_name="math accuracy", seed=0
    )
    print(f"verdict:          {verdict.verdict}")
    print(f"old accuracy:     {direct.scores.mean():.0%}  ({int(direct.scores.sum())}/{len(direct.scores)})")
    print(f"new accuracy:     {cot.scores.mean():.0%}  ({int(cot.scores.sum())}/{len(cot.scores)})")
    print(f"mean difference:  {verdict.mean_difference:+.4f}")
    print(f"95% CI:           ({verdict.ci_lower:+.4f}, {verdict.ci_upper:+.4f})")
    print(f"p-value:          {verdict.p_value:.4g}")
    print(f"method:           {verdict.method}")
    print(f"-> {verdict.recommendation}")

    # ========================================================================
    # Layer 2: explain() — stratification + driving examples
    # ========================================================================
    print()
    print("=" * 72)
    print("LAYER 2 — caliber.explain()  (where did the gain come from?)")
    print("=" * 72)
    explanation = caliber.explain(
        direct.scores, cot.scores,
        inputs=[p[0] for p in PROBLEMS],
        old_outputs=direct.responses,
        new_outputs=cot.responses,
        categories=[p[2] for p in PROBLEMS],
        top_n_examples=3,
        seed=0,
    )

    print()
    print("Stratified breakdown:")
    print(f"  {'category':<10}  {'n':>3}  {'direct':>7}  {'cot':>7}  {'delta':>7}")
    print("  " + "-" * 44)
    for s in sorted(explanation.strata, key=lambda x: -x.delta):
        print(f"  {s.name:<10}  {s.n:>3}  {s.old_accuracy:>6.0%}  {s.new_accuracy:>6.0%}  {s.delta:>+6.0%}")
    if explanation.biggest_gain_category and explanation.smallest_gain_category:
        print()
        print(f"  Biggest CoT benefit: {explanation.biggest_gain_category}")
        print(f"  Smallest CoT benefit: {explanation.smallest_gain_category}")

    print()
    print(f"Top {len(explanation.top_improvements)} driving improvements (where CoT flipped failure -> success):")
    for ex in explanation.top_improvements:
        cat = f"({ex.category})" if ex.category else ""
        q_short = ex.input if ex.input and len(ex.input) <= 50 else (ex.input[:47] + "..." if ex.input else "")
        print(f"  [#{ex.index:2d}] {cat:<10} {q_short}")
        print(f"        direct said: {ex.old_score:.0f} | cot said: {ex.new_score:.0f}  (delta {ex.delta:+.2f})")

    # ========================================================================
    # Layer 3: judge_hypothesis() — AI explains the pattern
    # ========================================================================
    print()
    print("=" * 72)
    print("LAYER 3 — caliber.judge_hypothesis()  (what's the pattern? phi3 reads the cases)")
    print("=" * 72)

    provider = caliber.OllamaProvider(model=MODEL, num_predict=400)
    print(f"\nAsking {provider.name}/{provider.model} to analyze "
          f"{min(3, len(explanation.top_improvements))} driving cases...")
    t0 = time.time()
    judged = caliber.judge_hypothesis(explanation, provider, n_examples=3)
    print(f"  elapsed: {time.time() - t0:.1f}s")

    print()
    print(f"Provider: {judged.provider}/{judged.model}")
    print(f"Cases reviewed: {judged.n_examples_reviewed}")
    print(f"AI-generated:  {judged.is_ai_generated}  (Caliber doesn't verify — your job)")
    print()
    print("Hypothesis:")
    print("-" * 72)
    print(judged.hypothesis)
    print("-" * 72)

    print()
    print("Done. Full pipeline complete: compare -> explain -> judge_hypothesis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
