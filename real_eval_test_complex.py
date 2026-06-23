"""Caliber end-to-end test against local phi3 — complex math problems.

Mix of:
  - order of operations (PEMDAS)
  - solve-for-x algebra
  - powers and roots
  - multi-step word problems
  - percentages, fractions, ratios

After scoring both prompts, prints phi3's full chain-of-thought response on a
handful of representative problems so you can see *how* the CoT prompt worked.

Run:
    python real_eval_test_complex.py
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

# 30 complex math problems. Each is (question, correct_integer_answer, category).
PROBLEMS: list[tuple[str, int, str]] = [
    # --- Order of operations (PEMDAS) ---
    ("Compute: 8 + 4 * 3 - 2",                                             18, "PEMDAS"),
    ("Compute: (12 + 6) * 4 - 10",                                         62, "PEMDAS"),
    ("Compute: 100 - 4^2 * 3",                                             52, "PEMDAS"),
    ("Compute: 5^2 + 3 * (8 - 4)",                                         37, "PEMDAS"),
    ("Compute: (15 - 7)^2 / 4 + 6",                                        22, "PEMDAS"),

    # --- Solve for x ---
    ("If 3x + 7 = 25, what is x?",                                          6, "algebra"),
    ("If 5x - 12 = 33, what is x?",                                         9, "algebra"),
    ("If 2(x + 4) = 26, what is x?",                                        9, "algebra"),
    ("If x/4 + 5 = 12, what is x?",                                        28, "algebra"),
    ("If 4x - 9 = 23, what is x?",                                          8, "algebra"),

    # --- Powers and roots ---
    ("What is 13 squared?",                                               169, "powers"),
    ("What is the square root of 256?",                                    16, "powers"),
    ("What is 5 to the power of 4?",                                      625, "powers"),
    ("What is 2 cubed plus 3 squared?",                                    17, "powers"),
    ("What is the cube root of 343?",                                       7, "powers"),

    # --- Multi-step word problems ---
    ("A car uses 4 gallons of gas to drive 100 miles. At the same rate, how many gallons are needed to drive 250 miles?", 10, "word"),
    ("A wallet has 7 five-dollar bills and 4 ten-dollar bills. What is the total value in dollars?", 75, "word"),
    ("The sum of three consecutive integers is 84. What is the smallest of the three?", 27, "word"),
    ("A rectangle has area 96 and length 12. What is its width?", 8, "word"),
    ("Train A leaves a station going 60 mph. Two hours later, Train B leaves the same station on the same track going 80 mph. How many hours after Train B leaves will it catch up to Train A?", 6, "word"),

    # --- Percentages and fractions ---
    ("What is 35 percent of 80?",                                          28, "percent"),
    ("If 45 percent of x equals 90, what is x?",                          200, "percent"),
    ("What do you get when you increase 60 by 25 percent?",                75, "percent"),
    ("What do you get when you decrease 200 by 15 percent?",              170, "percent"),
    ("What is three-eighths of 64?",                                       24, "percent"),

    # --- More mixed reasoning ---
    ("If 5 workers paint a house in 8 days, how many days would it take 4 workers working at the same rate?", 10, "word"),
    ("A book is 25 percent off the original price. After applying a 5 dollar coupon on top of the discount, the book costs 40 dollars. What was the original price in dollars?", 60, "word"),
    ("Two numbers sum to 20 and differ by 4. What is the larger number?",  12, "word"),
    ("A 100-liter tank is 60 percent full. How many more liters are needed to fill it completely?", 40, "word"),
    ("A square has perimeter 36. What is its area?",                       81, "word"),
]


PROMPT_DIRECT = (
    "{q}\n\n"
    "Reply with only the final number, no other text."
)

PROMPT_COT = (
    "{q}\n\n"
    "Think through this step by step. After your reasoning, write the final "
    "answer on its own line in the exact format: ANSWER = <number>"
)


def ask_phi3(prompt: str) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 600,
            },
        },
        timeout=180,
    )
    r.raise_for_status()
    return str(r.json()["response"])


def extract_number(text: str, prefer_after_label: bool = False) -> int | None:
    """Extract the model's numerical answer from its response."""
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


def score_one(prompt_template: str, question: str, correct: int, is_cot: bool) -> tuple[float, int | None, str]:
    response = ask_phi3(prompt_template.format(q=question))
    guess = extract_number(response, prefer_after_label=is_cot)
    return (1.0 if guess == correct else 0.0, guess, response)


def run_arm(prompt_template: str, label: str, is_cot: bool) -> tuple[np.ndarray, list[str], list[int | None]]:
    print(f"\n--- {label} ---")
    scores = np.zeros(len(PROBLEMS), dtype=np.float64)
    responses: list[str] = []
    guesses: list[int | None] = []
    t0 = time.time()
    for i, (question, correct, _category) in enumerate(PROBLEMS):
        score, guess, response = score_one(prompt_template, question, correct, is_cot)
        scores[i] = score
        responses.append(response)
        guesses.append(guess)
        status = "OK " if score == 1.0 else "BAD"
        guess_str = str(guess) if guess is not None else "?"
        q_short = question if len(question) <= 50 else question[:47] + "..."
        print(f"  [{i + 1:2d}/{len(PROBLEMS)}] {status}  correct={correct:5d}  phi3 said={guess_str:>6}  | {q_short}")
    dt = time.time() - t0
    print(f"  accuracy: {scores.mean():.2%}  ({int(scores.sum())}/{len(scores)})")
    print(f"  elapsed:  {dt:.1f}s ({dt / len(scores):.2f}s/problem)")
    return scores, responses, guesses


def show_cot_examples(
    responses: list[str],
    guesses: list[int | None],
    scores: np.ndarray,
    n_show: int = 3,
) -> None:
    """Print phi3's full CoT response for a handful of representative problems."""
    print()
    print("=" * 70)
    print(f"How the chain-of-thought prompt actually worked — {n_show} examples")
    print("=" * 70)

    # Pick 2 successes from different categories + 1 failure (if any).
    successes = [i for i in range(len(PROBLEMS)) if scores[i] == 1.0]
    failures  = [i for i in range(len(PROBLEMS)) if scores[i] == 0.0]
    picks: list[int] = []
    seen_categories: set[str] = set()
    for i in successes:
        cat = PROBLEMS[i][2]
        if cat not in seen_categories:
            picks.append(i)
            seen_categories.add(cat)
        if len(picks) >= n_show - 1:
            break
    if failures:
        picks.append(failures[0])

    for i in picks[:n_show]:
        question, correct, category = PROBLEMS[i]
        guess = guesses[i]
        verdict_str = "CORRECT" if scores[i] == 1.0 else "WRONG"
        print()
        print(f"--- Problem {i + 1} ({category}) — {verdict_str} ---")
        print(f"Question:        {question}")
        print(f"Correct answer:  {correct}")
        print(f"phi3 answered:   {guess}")
        print()
        print("Full chain-of-thought response from phi3:")
        print("-" * 70)
        print(responses[i].strip())
        print("-" * 70)


def main() -> int:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: Ollama not reachable at localhost:11434 ({e})", file=sys.stderr)
        return 1

    print(f"Caliber end-to-end test — local {MODEL} via Ollama")
    print(f"Task: {len(PROBLEMS)} complex math problems "
          "(PEMDAS, algebra, powers, word, percentages)")
    print(f"Comparing: direct-answer vs chain-of-thought")

    old_scores, _, _ = run_arm(PROMPT_DIRECT, "OLD: direct answer", is_cot=False)
    new_scores, new_responses, new_guesses = run_arm(PROMPT_COT, "NEW: chain-of-thought", is_cot=True)

    print()
    print("=" * 70)
    print("Caliber verdict")
    print("=" * 70)
    result = caliber.compare(old_scores, new_scores, metric_name="math accuracy", seed=0)
    print(f"verdict:          {result.verdict}")
    print(f"old accuracy:     {old_scores.mean():.2%}  ({int(old_scores.sum())}/{len(old_scores)})")
    print(f"new accuracy:     {new_scores.mean():.2%}  ({int(new_scores.sum())}/{len(new_scores)})")
    print(f"mean difference:  {result.mean_difference:+.4f}")
    print(f"95% CI:           ({result.ci_lower:+.4f}, {result.ci_upper:+.4f})")
    print(f"p-value:          {result.p_value:.4g}")
    print(f"method:           {result.method}")
    print(f"n pairs:          {result.n}")
    print()
    print(result.recommendation)

    # Show how the CoT prompt actually worked on a few problems.
    show_cot_examples(new_responses, new_guesses, new_scores, n_show=3)

    return 0


if __name__ == "__main__":
    sys.exit(main())
