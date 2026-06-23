"""Test: two GOOD prompts on logic/reasoning problems.

Both prompts use chain-of-thought — this is not a CoT-vs-no-CoT setup.
The two prompts differ on a real prompt-engineering choice:

  Prompt 1 — Freeform CoT.    "Think step by step, then answer."
  Prompt 2 — Structured CoT.  "List the GIVEN facts, state what to FIND,
                               then SOLVE."

Both are legitimate. The question is: does the explicit structure
in Prompt 2 actually help on logic/reasoning problems, or does it just
add noise? Caliber decides.

Run:
    python -X utf8 real_eval_two_prompts.py
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

# 10 logic / reasoning problems with single-word or numeric answers.
PROBLEMS: list[tuple[str, str, str]] = [
    # --- Syllogisms ---
    ("All mammals have hair. Whales are mammals. Do whales have hair? Reply with only 'yes' or 'no'.",
     "yes", "syllogism"),
    ("All birds lay eggs. Some animals that lay eggs are reptiles. Therefore some birds are reptiles. "
     "Is this argument valid? Reply with only 'yes' or 'no'.",
     "no", "syllogism"),
    # --- Spatial / ordering ---
    ("Alice is taller than Bob. Bob is taller than Carol. Who is the shortest? Reply with just the name.",
     "Carol", "spatial"),
    ("Five books are stacked from bottom to top: Math, Science, History, Art, Biology. "
     "Which book is third from the bottom? Reply with just the subject name.",
     "History", "spatial"),
    # --- Counting / combinatorics ---
    ("There are 5 people in a room. Each shakes hands with every other person exactly once. "
     "How many handshakes total?",
     "10", "counting"),
    ("How many distinct 2-letter sequences can be formed from the letters A, B, C if letters can repeat?",
     "9", "counting"),
    # --- Sequences ---
    ("What comes next in this sequence: 1, 4, 9, 16, ?",
     "25", "sequence"),
    ("What number is missing: 3, 7, 11, ?, 19, 23",
     "15", "sequence"),
    # --- Conditional / contrapositive ---
    ("If it rains, the streets get wet. The streets are not wet. Did it rain? Reply with only 'yes' or 'no'.",
     "no", "conditional"),
    ("If a number is divisible by 4, it is also divisible by 2. The number 6 is divisible by 2. "
     "Is 6 divisible by 4? Reply with only 'yes' or 'no'.",
     "no", "conditional"),
]


# Both prompts force CoT. They differ in HOW the chain of thought is structured.
PROMPT_1_FREEFORM = (
    "{q}\n\n"
    "Think through this carefully and step by step. After your reasoning, "
    "write the final answer on its own line in the exact format: ANSWER = <value>"
)

PROMPT_2_STRUCTURED = (
    "{q}\n\n"
    "Solve this in three explicit steps:\n"
    "1. GIVEN: List the facts stated in the problem.\n"
    "2. FIND: State exactly what you need to determine.\n"
    "3. SOLVE: Reason through it to reach the answer.\n\n"
    "After your reasoning, write the final answer on its own line in the "
    "exact format: ANSWER = <value>"
)


def ask_phi3(prompt: str) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 700},
        },
        timeout=180,
    )
    r.raise_for_status()
    return str(r.json()["response"])


def _normalize(s: str) -> str:
    """Lowercase + strip non-alphanumeric (keeps digits intact)."""
    return re.sub(r"[^\w]", "", str(s).lower())


def extract_answer(text: str) -> str:
    """Pull the model's stated answer out of its response.

    Looks for 'ANSWER = X' first; falls back to the last word in the response.
    """
    m = re.search(r"answer\s*[:=]\s*([^\n]+)", text, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().rstrip(".,;:")
        first = candidate.split()[0] if candidate else ""
        return _normalize(first)
    # Fallback: last word
    words = text.strip().split()
    return _normalize(words[-1]) if words else ""


def is_correct(extracted: str, expected: str) -> bool:
    return extracted == _normalize(expected)


@dataclass
class ArmResult:
    scores: np.ndarray
    responses: list[str]
    guesses: list[str]


def run_arm(template: str, label: str) -> ArmResult:
    print(f"\n--- {label} ---")
    scores = np.zeros(len(PROBLEMS), dtype=np.float64)
    responses: list[str] = []
    guesses: list[str] = []
    t0 = time.time()
    for i, (q, correct, _cat) in enumerate(PROBLEMS):
        response = ask_phi3(template.format(q=q))
        guess = extract_answer(response)
        score = 1.0 if is_correct(guess, correct) else 0.0
        scores[i] = score
        responses.append(response)
        guesses.append(guess)
        status = "OK " if score else "BAD"
        q_short = q if len(q) <= 50 else q[:47] + "..."
        print(f"  [{i + 1:2d}/{len(PROBLEMS)}] {status}  correct={correct:>8}  phi3 said={guess:>8}  | {q_short}")
    print(f"  accuracy: {scores.mean():.0%}  elapsed: {time.time() - t0:.1f}s")
    return ArmResult(scores, responses, guesses)


def main() -> int:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: Ollama not reachable ({e})", file=sys.stderr)
        return 1

    print(f"Caliber two-prompt comparison — {MODEL} via Ollama")
    print(f"Task: {len(PROBLEMS)} logic/reasoning problems")
    print(f"Prompt 1: Freeform 'think step by step' CoT")
    print(f"Prompt 2: Structured 'GIVEN / FIND / SOLVE' CoT")
    print(f"Pipeline: compare() -> explain() -> judge_hypothesis()")

    arm1 = run_arm(PROMPT_1_FREEFORM,   "Prompt 1: Freeform CoT")
    arm2 = run_arm(PROMPT_2_STRUCTURED, "Prompt 2: Structured CoT")

    # ========================================================================
    # Layer 1 — verdict
    # ========================================================================
    print()
    print("=" * 72)
    print("LAYER 1 — caliber.compare()  (which prompt is statistically better?)")
    print("=" * 72)
    # NOTE: arm1 -> old_scores, arm2 -> new_scores, so verdict BETTER means
    # Prompt 2 is better than Prompt 1.
    verdict = caliber.compare(
        arm1.scores, arm2.scores, metric_name="logic accuracy", seed=0
    )
    print(f"verdict (Prompt 2 vs Prompt 1):  {verdict.verdict}")
    print(f"Prompt 1 accuracy:  {arm1.scores.mean():.0%}  ({int(arm1.scores.sum())}/{len(arm1.scores)})")
    print(f"Prompt 2 accuracy:  {arm2.scores.mean():.0%}  ({int(arm2.scores.sum())}/{len(arm2.scores)})")
    print(f"mean difference:    {verdict.mean_difference:+.4f}  (positive = Prompt 2 better)")
    print(f"95% CI:             ({verdict.ci_lower:+.4f}, {verdict.ci_upper:+.4f})")
    print(f"p-value:            {verdict.p_value:.4g}")
    print(f"method:             {verdict.method}")
    print()

    # Translate the verdict into prompt-1-vs-prompt-2 language
    if verdict.verdict == "BETTER":
        winner = "Prompt 2 (Structured)"
        loser = "Prompt 1 (Freeform)"
    elif verdict.verdict == "WORSE":
        winner = "Prompt 1 (Freeform)"
        loser = "Prompt 2 (Structured)"
    elif verdict.verdict == "NO_CHANGE":
        winner = None
        loser = None
        print("Verdict: NO_CHANGE — the two prompts produce essentially the same accuracy.")
    else:  # INCONCLUSIVE
        winner = None
        loser = None
        print("Verdict: INCONCLUSIVE — there's not enough evidence at n="
              f"{len(arm1.scores)} to call one prompt better than the other.")

    if winner:
        print(f">>> WINNER:  {winner}")
        print(f">>> LOSER:   {loser}")

    # ========================================================================
    # Layer 2 — explain (where did the difference come from?)
    # ========================================================================
    print()
    print("=" * 72)
    print("LAYER 2 — caliber.explain()  (where did the difference come from?)")
    print("=" * 72)
    explanation = caliber.explain(
        arm1.scores, arm2.scores,
        inputs=[p[0] for p in PROBLEMS],
        old_outputs=arm1.responses,
        new_outputs=arm2.responses,
        categories=[p[2] for p in PROBLEMS],
        top_n_examples=3,
        seed=0,
    )

    print()
    print("Stratified breakdown:")
    print(f"  {'category':<12}  {'n':>3}  {'P1':>6}  {'P2':>6}  {'delta':>7}")
    print("  " + "-" * 44)
    for s in sorted(explanation.strata, key=lambda x: -x.delta):
        print(f"  {s.name:<12}  {s.n:>3}  {s.old_accuracy:>5.0%}  {s.new_accuracy:>5.0%}  {s.delta:>+6.0%}")
    if explanation.biggest_gain_category and explanation.smallest_gain_category:
        print()
        if explanation.biggest_gain_category != explanation.smallest_gain_category:
            print(f"  Where Prompt 2 helped most:   {explanation.biggest_gain_category}")
            print(f"  Where Prompt 2 helped least:  {explanation.smallest_gain_category}")
        else:
            print(f"  Effect roughly equal across categories.")

    if explanation.top_improvements:
        print()
        print(f"Top {len(explanation.top_improvements)} cases where Prompt 2 beat Prompt 1:")
        for ex in explanation.top_improvements:
            cat = f"({ex.category})" if ex.category else ""
            q_short = ex.input if ex.input and len(ex.input) <= 60 else (ex.input[:57] + "..." if ex.input else "")
            print(f"  [#{ex.index:2d}] {cat:<12}  {q_short}")
            print(f"            P1 said: {ex.old_score:.0f}  | P2 said: {ex.new_score:.0f}")
    if explanation.top_regressions:
        print()
        print(f"Top {len(explanation.top_regressions)} cases where Prompt 1 beat Prompt 2:")
        for ex in explanation.top_regressions:
            cat = f"({ex.category})" if ex.category else ""
            q_short = ex.input if ex.input and len(ex.input) <= 60 else (ex.input[:57] + "..." if ex.input else "")
            print(f"  [#{ex.index:2d}] {cat:<12}  {q_short}")
            print(f"            P1 said: {ex.old_score:.0f}  | P2 said: {ex.new_score:.0f}")

    # ========================================================================
    # Layer 3 — judge_hypothesis (why?)
    # ========================================================================
    print()
    print("=" * 72)
    print("LAYER 3 — caliber.judge_hypothesis()  (why is the winner better?)")
    print("=" * 72)

    # If verdict is BETTER or WORSE, surface a hypothesis using whichever
    # direction has driving examples. If INCONCLUSIVE/NO_CHANGE, skip.
    if verdict.verdict in ("BETTER", "WORSE"):
        # The judge layer reads explanation.top_improvements (Prompt 2 wins).
        # If Prompt 1 won (WORSE verdict), the wins-for-2 list might be empty
        # while the regressions list (Prompt 1 wins) is populated. Skip judge
        # if there aren't at least 2 cases for it to read.
        winning_examples = (
            explanation.top_improvements if verdict.verdict == "BETTER"
            else explanation.top_regressions
        )
        if len(winning_examples) >= 2:
            provider = caliber.OllamaProvider(model=MODEL, num_predict=400)
            print(f"\nAsking {provider.name}/{provider.model} to analyze "
                  f"{min(3, len(winning_examples))} cases where the winner outperformed...")
            t0 = time.time()
            # The judge reads the explanation's top_improvements (Prompt 2 wins).
            # When verdict is WORSE, we'd need a different approach to ask
            # about Prompt 1's wins; for v1 we only surface when Prompt 2 won.
            if verdict.verdict == "BETTER":
                judged = caliber.judge_hypothesis(explanation, provider, n_examples=3)
                print(f"  elapsed: {time.time() - t0:.1f}s")
                print()
                print(f"Provider:    {judged.provider}/{judged.model}")
                print(f"Reviewed:    {judged.n_examples_reviewed} cases")
                print(f"AI-generated (Caliber doesn't verify):  {judged.is_ai_generated}")
                print()
                print("Hypothesis — why Prompt 2 was better:")
                print("-" * 72)
                print(judged.hypothesis)
                print("-" * 72)
            else:
                # Prompt 1 won — invert the explanation by swapping arms
                inverted = caliber.explain(
                    arm2.scores, arm1.scores,
                    inputs=[p[0] for p in PROBLEMS],
                    old_outputs=arm2.responses,
                    new_outputs=arm1.responses,
                    categories=[p[2] for p in PROBLEMS],
                    top_n_examples=3,
                    seed=0,
                )
                judged = caliber.judge_hypothesis(inverted, provider, n_examples=3)
                print(f"  elapsed: {time.time() - t0:.1f}s")
                print()
                print(f"Provider:    {judged.provider}/{judged.model}")
                print(f"Reviewed:    {judged.n_examples_reviewed} cases")
                print(f"AI-generated (Caliber doesn't verify):  {judged.is_ai_generated}")
                print()
                print("Hypothesis — why Prompt 1 was better:")
                print("-" * 72)
                print(judged.hypothesis)
                print("-" * 72)
        else:
            print("\nNot enough winning examples to ask the LLM judge (need >=2).")
    else:
        print(f"\nSkipping hypothesis — verdict is {verdict.verdict}, "
              "no winner to explain.")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
