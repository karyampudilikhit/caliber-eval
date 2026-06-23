"""Prototype of Caliber's 'explain' feature.

After computing the standard verdict, we ALSO:
  Tier 1 — stratify scores by category (where did the gain come from?)
  Tier 2 — surface the specific examples that flipped (driving cases)
  Tier 3 — ask an LLM judge to articulate the pattern across those flips

Tiers 1 and 2 are pure data analysis. Tier 3 uses phi3 itself as a judge
to read the win/loss responses and propose why the new prompt is better.

Run:
    python real_eval_with_explain.py
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

# 10 problems across 5 categories so stratification has something to show.
PROBLEMS: list[tuple[str, int, str]] = [
    ("Compute: 8 + 4 * 3 - 2",                                             18, "PEMDAS"),
    ("Compute: (12 + 6) * 4 - 10",                                         62, "PEMDAS"),
    ("If 3x + 7 = 25, what is x?",                                          6, "algebra"),
    ("If 5x - 12 = 33, what is x?",                                         9, "algebra"),
    ("What is 13 squared?",                                               169, "powers"),
    ("What is the cube root of 343?",                                       7, "powers"),
    ("The sum of three consecutive integers is 84. What is the smallest?", 27, "word"),
    ("Two numbers sum to 20 and differ by 4. What is the larger?",         12, "word"),
    ("What is 35 percent of 80?",                                          28, "percent"),
    ("Increase 60 by 25 percent.",                                         75, "percent"),
]


PROMPT_DIRECT = "{q}\n\nReply with only the final number, no other text."
PROMPT_COT = (
    "{q}\n\nThink through this step by step. After your reasoning, write the "
    "final answer on its own line in the exact format: ANSWER = <number>"
)


def ask_phi3(prompt: str, num_predict: int = 600) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": num_predict},
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


# ----------------------------------------------------------------------------
# Run both arms — collect scores AND responses for the explain step
# ----------------------------------------------------------------------------


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
        print(f"  [{i + 1:2d}/{len(PROBLEMS)}] {status}  correct={correct:4d}  phi3 said={str(guess):>6}  | {q_short}")
    print(f"  accuracy: {scores.mean():.0%}  elapsed: {time.time() - t0:.1f}s")
    return ArmResult(scores, responses, guesses)


# ----------------------------------------------------------------------------
# Tier 1 — stratify by category
# ----------------------------------------------------------------------------


def tier1_stratify(direct: ArmResult, cot: ArmResult) -> None:
    print()
    print("=" * 70)
    print("Tier 1 — Stratified analysis (where did the gain come from?)")
    print("=" * 70)
    categories = sorted({p[2] for p in PROBLEMS})
    print(f"{'category':<12}  {'n':>3}  {'direct':>8}  {'cot':>8}  {'delta':>8}")
    print("-" * 50)
    for cat in categories:
        idx = [i for i, p in enumerate(PROBLEMS) if p[2] == cat]
        d = float(direct.scores[idx].mean())
        c = float(cot.scores[idx].mean())
        print(f"{cat:<12}  {len(idx):>3}  {d:>7.0%}  {c:>7.0%}  {c - d:>+7.0%}")
    print()
    # Identify largest-delta category for the takeaway
    deltas_by_cat = [
        (cat, float(cot.scores[[i for i, p in enumerate(PROBLEMS) if p[2] == cat]].mean()
                    - direct.scores[[i for i, p in enumerate(PROBLEMS) if p[2] == cat]].mean()))
        for cat in categories
    ]
    deltas_by_cat.sort(key=lambda x: x[1], reverse=True)
    biggest = deltas_by_cat[0]
    smallest = deltas_by_cat[-1]
    print(f"Largest CoT benefit:  {biggest[0]:<10} (+{biggest[1]:.0%})")
    print(f"Smallest CoT benefit: {smallest[0]:<10} (+{smallest[1]:.0%})")


# ----------------------------------------------------------------------------
# Tier 2 — surface the specific flipped examples
# ----------------------------------------------------------------------------


def tier2_driving_examples(direct: ArmResult, cot: ArmResult) -> list[int]:
    print()
    print("=" * 70)
    print("Tier 2 — Driving examples (where did the new prompt actually flip?)")
    print("=" * 70)
    flips_to_win = [
        i for i in range(len(PROBLEMS))
        if direct.scores[i] == 0 and cot.scores[i] == 1
    ]
    flips_to_lose = [
        i for i in range(len(PROBLEMS))
        if direct.scores[i] == 1 and cot.scores[i] == 0
    ]
    print(f"{len(flips_to_win)} cases where CoT flipped a failure to success:")
    for i in flips_to_win[:3]:
        q = PROBLEMS[i][0]
        print(f"  [{i + 1}] ({PROBLEMS[i][2]}) {q}")
        print(f"      direct said: {direct.guesses[i]}  ❌")
        print(f"      cot said:    {cot.guesses[i]}  ✓ (correct: {PROBLEMS[i][1]})")
    if flips_to_lose:
        print(f"\n{len(flips_to_lose)} regressions (CoT broke what direct got right):")
        for i in flips_to_lose:
            q = PROBLEMS[i][0]
            print(f"  [{i + 1}] ({PROBLEMS[i][2]}) {q}")
            print(f"      direct said: {direct.guesses[i]}  ✓")
            print(f"      cot said:    {cot.guesses[i]}  ❌ (correct: {PROBLEMS[i][1]})")
    return flips_to_win


# ----------------------------------------------------------------------------
# Tier 3 — LLM-judged hypothesis for why
# ----------------------------------------------------------------------------


def tier3_llm_hypothesis(direct: ArmResult, cot: ArmResult, flips_to_win: list[int]) -> None:
    print()
    print("=" * 70)
    print("Tier 3 — LLM-judged hypothesis (what's the pattern across the flips?)")
    print("=" * 70)
    if len(flips_to_win) < 2:
        print("Not enough flipped cases to ask the judge.")
        return

    # Build a meta-prompt: show the judge the inputs and BOTH responses for
    # 3 flipped cases, then ask "what's the pattern?".
    sample = flips_to_win[:3]
    judge_prompt_parts = [
        "You are a prompt-engineering analyst. Below are 3 cases where prompt A "
        "produced a wrong answer but prompt B produced a correct answer on the "
        "same math problem. Identify the pattern in 1-2 sentences: what does "
        "prompt B do differently that helps the model get the right answer?",
        "",
    ]
    for n, i in enumerate(sample, start=1):
        q, correct, _ = PROBLEMS[i]
        a_response = direct.responses[i].strip()
        b_response = cot.responses[i].strip()
        # Truncate very long CoT responses
        b_response = (b_response[:800] + "...") if len(b_response) > 800 else b_response
        judge_prompt_parts.extend([
            f"=== Case {n} ===",
            f"Question: {q}",
            f"Correct answer: {correct}",
            "",
            f"Prompt A response (WRONG):",
            a_response,
            "",
            f"Prompt B response (CORRECT):",
            b_response,
            "",
        ])
    judge_prompt_parts.append("Pattern (1-2 sentences):")
    judge_prompt = "\n".join(judge_prompt_parts)

    print(f"Asking phi3 to analyze {len(sample)} flipped cases...")
    print()
    hypothesis = ask_phi3(judge_prompt, num_predict=400)
    print("phi3's hypothesis:")
    print("-" * 70)
    print(hypothesis.strip())
    print("-" * 70)
    print()
    print("(Hypothesis is AI-generated. Caliber doesn't verify it — that's your job.)")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: Ollama not reachable ({e})", file=sys.stderr)
        return 1

    print(f"Caliber + explain prototype — {MODEL} via Ollama")
    print(f"Task: {len(PROBLEMS)} problems across "
          f"{len(set(p[2] for p in PROBLEMS))} categories")

    direct = run_arm(PROMPT_DIRECT, "OLD: direct answer", is_cot=False)
    cot = run_arm(PROMPT_COT, "NEW: chain-of-thought", is_cot=True)

    # Standard Caliber verdict
    print()
    print("=" * 70)
    print("Caliber verdict (statistics)")
    print("=" * 70)
    result = caliber.compare(direct.scores, cot.scores, metric_name="math accuracy", seed=0)
    print(f"verdict:          {result.verdict}")
    print(f"old accuracy:     {direct.scores.mean():.0%}")
    print(f"new accuracy:     {cot.scores.mean():.0%}")
    print(f"mean difference:  {result.mean_difference:+.4f}")
    print(f"95% CI:           ({result.ci_lower:+.4f}, {result.ci_upper:+.4f})")
    print(f"p-value:          {result.p_value:.4g}")

    # The new "why" layer
    tier1_stratify(direct, cot)
    flips = tier2_driving_examples(direct, cot)
    tier3_llm_hypothesis(direct, cot, flips)

    return 0


if __name__ == "__main__":
    sys.exit(main())
