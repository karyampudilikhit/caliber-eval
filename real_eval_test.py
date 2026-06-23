"""Real end-to-end test of Caliber against a local LLM (phi3 via Ollama).

Hypothesis: chain-of-thought prompting helps small models on multi-step
arithmetic. We test two prompts on 50 two-digit multiplication problems
and ask Caliber whether the difference is statistically real.

Why this setup:
- phi3 is small (3.8B) → weak at mental multiplication, so CoT should help
- multiplication has a deterministic correct answer → no judge model needed
- 50 problems × 2 prompts × ~few seconds each = a few minutes to run locally

Run:
    python real_eval_test.py
"""

from __future__ import annotations

import re
import sys
import time

import numpy as np
import requests

import caliber

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3"

# 30 two-digit multiplication problems with their correct answers.
# Generated deterministically so the eval is reproducible.
_rng = np.random.default_rng(0)
PROBLEMS: list[tuple[int, int, int]] = []
while len(PROBLEMS) < 30:
    a = int(_rng.integers(11, 100))
    b = int(_rng.integers(11, 100))
    PROBLEMS.append((a, b, a * b))


PROMPT_DIRECT = (
    "What is {a} times {b}? "
    "Reply with only the number, nothing else."
)

PROMPT_COT = (
    "What is {a} times {b}? "
    "Work it out step by step, then on the final line write "
    "exactly: Answer: <number>"
)


def ask_phi3(prompt: str) -> str:
    """One round-trip to the local Ollama server."""
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,   # deterministic — we want a clean test
                "num_predict": 500,   # enough for CoT to actually finish
            },
        },
        timeout=120,
    )
    r.raise_for_status()
    return str(r.json()["response"])


def extract_number(text: str, prefer_after_label: bool = False) -> int | None:
    """Pull a number out of the model's response.

    If prefer_after_label is True (CoT path), prefer the number after
    'Answer:' if one exists; otherwise fall back to the last number in
    the text.
    """
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


def score_one(prompt_template: str, a: int, b: int, correct: int, is_cot: bool) -> float:
    response = ask_phi3(prompt_template.format(a=a, b=b))
    guess = extract_number(response, prefer_after_label=is_cot)
    return 1.0 if guess == correct else 0.0


def run_arm(prompt_template: str, label: str, is_cot: bool) -> np.ndarray:
    print(f"\n--- {label} ---")
    scores = np.zeros(len(PROBLEMS), dtype=np.float64)
    t0 = time.time()
    for i, (a, b, correct) in enumerate(PROBLEMS):
        scores[i] = score_one(prompt_template, a, b, correct, is_cot)
        status = "OK " if scores[i] == 1.0 else "BAD"
        print(f"  [{i + 1:2d}/{len(PROBLEMS)}] {a:2d} x {b:2d} = {correct:5d}  {status}")
    dt = time.time() - t0
    print(f"  accuracy: {scores.mean():.2%}  ({int(scores.sum())}/{len(scores)})")
    print(f"  elapsed:  {dt:.1f}s ({dt / len(scores):.2f}s/problem)")
    return scores


def main() -> int:
    # Sanity check Ollama is reachable.
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: Ollama not reachable at localhost:11434 ({e})", file=sys.stderr)
        return 1

    print(f"Running Caliber end-to-end test against local {MODEL} via Ollama.")
    print(f"Task: 50 two-digit multiplication problems.")
    print(f"Comparing: direct-answer prompt vs chain-of-thought prompt.")

    old_scores = run_arm(PROMPT_DIRECT, "OLD: direct answer", is_cot=False)
    new_scores = run_arm(PROMPT_COT,    "NEW: chain-of-thought", is_cot=True)

    print()
    print("=" * 60)
    print("Caliber verdict")
    print("=" * 60)
    result = caliber.compare(old_scores, new_scores, metric_name="multiplication accuracy")
    print(f"verdict:          {result.verdict}")
    print(f"old accuracy:     {old_scores.mean():.2%}")
    print(f"new accuracy:     {new_scores.mean():.2%}")
    print(f"mean difference:  {result.mean_difference:+.4f}")
    print(f"95% CI:           ({result.ci_lower:+.4f}, {result.ci_upper:+.4f})")
    print(f"p-value:          {result.p_value:.4g}")
    print(f"method:           {result.method}")
    print(f"n pairs:          {result.n}")
    print()
    print(result.recommendation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
