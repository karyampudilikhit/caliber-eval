"""Caliber end-to-end test against local phi3 — multi-step word problems.

Harder than pure arithmetic: each problem requires reading comprehension,
identifying the right operations, and chaining them in the right order.
This is where CoT prompting genuinely matters for small models.

Run:
    python real_eval_test_words.py
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

# 30 multi-step word problems with integer answers.
# Each tuple is (question, correct_answer).
PROBLEMS: list[tuple[str, int]] = [
    ("Tom has 24 marbles. He gives 8 to his friend and finds 5 more. How many marbles does Tom have now?", 21),
    ("A baker makes 36 cookies. He sells 12 in the morning and 18 in the afternoon. How many cookies are left?", 6),
    ("Sarah read 45 pages on Monday, 32 on Tuesday, and 28 on Wednesday. How many pages did she read in total?", 105),
    ("Mike has 50 dollars. He buys 3 books at 8 dollars each. How much money does he have left in dollars?", 26),
    ("A classroom has 24 students. If one-third of them are absent today, how many students are present?", 16),
    ("Lisa is 12 years old. Her brother is 4 years younger than her. Their mother is 5 times as old as Lisa's brother. How old is the mother?", 40),
    ("A rectangle has length 15 and width 8. What is its perimeter?", 46),
    ("Anna walks 3 km north, then 4 km east, then 3 km south. How far is she from her starting point in km?", 4),
    ("A movie ticket costs 12 dollars for adults and 7 dollars for children. A family buys 2 adult and 3 child tickets. What is the total cost in dollars?", 45),
    ("John has twice as many apples as oranges. He has 5 oranges. He eats 3 apples. How many apples does he have left?", 7),
    ("A train leaves at 2:30 PM and arrives at 5:15 PM. How long was the trip in minutes?", 165),
    ("Maria saves 25 dollars each week. After 8 weeks, she spends 80 dollars on a gift. How many dollars does she have left?", 120),
    ("Tom is 10 years older than Jerry. In 5 years, Tom will be twice as old as Jerry. How old is Jerry now?", 5),
    ("Sam has a 50-meter rope. He cuts off 12 pieces of 3 meters each. How much rope is left in meters?", 14),
    ("A book has 240 pages. Tim reads 30 pages a day. After 5 days, how many pages are left?", 90),
    ("A school has 5 classrooms with 28 students each, plus 1 extra room with 20 students. How many students in total?", 160),
    ("There are 60 students. 25 percent play soccer and 35 percent play basketball, with no overlap. How many students play neither sport?", 24),
    ("A car drives at 65 mph for 3 hours, then stops for 1 hour, then drives at 50 mph for 2 hours. Total distance in miles?", 295),
    ("Linda has 200 dollars. She gives one-quarter to her sister and one-fifth to her brother. How many dollars does Linda have left?", 110),
    ("A water tank starts with 80 gallons. It is filled at 5 gallons per hour and drained at 3 gallons per hour. After 6 hours, how many gallons are in the tank?", 92),
    ("Bob worked 8 hours on Monday and 6 hours on Tuesday at 15 dollars per hour. How many dollars did he earn in total?", 210),
    ("A box has 36 chocolates. Two-thirds are dark and the rest are milk. How many milk chocolates are there?", 12),
    ("Karen bought 4 shirts at 25 dollars each. The store gave her a 20 dollar discount. How many dollars did she pay?", 80),
    ("A class has 18 boys and 22 girls. Then 3 boys and 4 girls leave. How many students are left?", 33),
    ("Tim is 25 years old. His sister is 7 years younger. Their father is the sum of their ages plus 5. How old is the father?", 48),
    ("A garden is 12 meters long and 9 meters wide. What is its area in square meters?", 108),
    ("A taxi charges 5 dollars base fare plus 2 dollars per mile. How many dollars does a 15-mile ride cost?", 35),
    ("Lucy has 60 stickers. She gives one-third to Mark and one-quarter of the original number to Jane. How many stickers does Lucy have left?", 25),
    ("A cinema has 12 rows with 25 seats each. If 240 people are watching a movie, how many seats are empty?", 60),
    ("A factory makes 144 widgets in 6 hours. At the same rate, how many widgets can it make in 9 hours?", 216),
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
        # Strong signal: explicit ANSWER = N marker.
        m = re.search(r"answer\s*[:=]\s*(-?\d[\d,]*)", text, re.IGNORECASE)
        if m:
            return int(m.group(1).replace(",", ""))
    # Otherwise take the last number in the response.
    nums = re.findall(r"-?\d[\d,]*", text)
    if not nums:
        return None
    try:
        return int(nums[-1].replace(",", ""))
    except ValueError:
        return None


def score_one(prompt_template: str, question: str, correct: int, is_cot: bool) -> tuple[float, int | None]:
    response = ask_phi3(prompt_template.format(q=question))
    guess = extract_number(response, prefer_after_label=is_cot)
    return (1.0 if guess == correct else 0.0, guess)


def run_arm(prompt_template: str, label: str, is_cot: bool) -> np.ndarray:
    print(f"\n--- {label} ---")
    scores = np.zeros(len(PROBLEMS), dtype=np.float64)
    t0 = time.time()
    for i, (question, correct) in enumerate(PROBLEMS):
        score, guess = score_one(prompt_template, question, correct, is_cot)
        scores[i] = score
        status = "OK " if score == 1.0 else "BAD"
        guess_str = f"guess={guess}" if guess is not None else "no_number"
        # Truncate question for log readability
        q_short = question if len(question) <= 60 else question[:57] + "..."
        print(f"  [{i + 1:2d}/{len(PROBLEMS)}] {status}  correct={correct:4d}  {guess_str:>15}  | {q_short}")
    dt = time.time() - t0
    print(f"  accuracy: {scores.mean():.2%}  ({int(scores.sum())}/{len(scores)})")
    print(f"  elapsed:  {dt:.1f}s ({dt / len(scores):.2f}s/problem)")
    return scores


def main() -> int:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: Ollama not reachable at localhost:11434 ({e})", file=sys.stderr)
        return 1

    print(f"Caliber end-to-end test — local {MODEL} via Ollama")
    print(f"Task: {len(PROBLEMS)} multi-step word problems")
    print(f"Comparing: direct-answer vs chain-of-thought")

    old_scores = run_arm(PROMPT_DIRECT, "OLD: direct answer", is_cot=False)
    new_scores = run_arm(PROMPT_COT,    "NEW: chain-of-thought", is_cot=True)

    print()
    print("=" * 70)
    print("Caliber verdict")
    print("=" * 70)
    result = caliber.compare(old_scores, new_scores, metric_name="word-problem accuracy", seed=0)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
