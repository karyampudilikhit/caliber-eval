"""Test: Prompt 1 vs Prompt 2 on 100 logic/reasoning problems.

Same prompt pair as real_eval_two_prompts_v2.py — but problems are now generated
from templates with seeded randomness so we can scale to large n.

At n=100, Caliber's sample_size analysis says we can detect a true 10pp gap
with 80% power. So if there's a real difference here, this test will surface
it. If the verdict is still INCONCLUSIVE at n=100, the honest finding is that
these two prompts are statistically equivalent for phi3 on these problem types.

Run:
    python -X utf8 real_eval_n100.py
"""

from __future__ import annotations

import random
import re
import sys
import time
from dataclasses import dataclass

import numpy as np
import requests

import caliber

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3"

# ============================================================================
# Problem generation — 20 problems per category, 5 categories, total 100
# ============================================================================


def generate_problems(seed: int = 42) -> list[tuple[str, str, str]]:
    rng = random.Random(seed)
    problems: list[tuple[str, str, str]] = []
    problems.extend(_gen_spatial(rng, 20))
    problems.extend(_gen_counting(rng, 20))
    problems.extend(_gen_sequence(rng, 20))
    problems.extend(_gen_conditional(rng, 20))
    problems.extend(_gen_syllogism(rng, 20))
    rng.shuffle(problems)
    return problems


def _gen_spatial(rng: random.Random, k: int) -> list[tuple[str, str, str]]:
    names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
             "Ivy", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia"]
    relations = ["taller", "older", "faster", "richer", "heavier"]
    out: list[tuple[str, str, str]] = []
    for i in range(k):
        kind = i % 4
        if kind == 0:
            # 3-way chain — who is least?
            n = rng.sample(names, 3)
            rel = rng.choice(relations)
            out.append((
                f"{n[0]} is {rel} than {n[1]}. {n[1]} is {rel} than {n[2]}. "
                f"Who is the least {rel}? Reply with just the name.",
                n[2], "spatial",
            ))
        elif kind == 1:
            # Position in a line
            picks = rng.sample(names, 4)
            target_pos = rng.choice([1, 2, 3])  # 0-indexed -> position 2, 3, 4 from left
            ordinal = {1: "second", 2: "third", 3: "fourth"}[target_pos]
            out.append((
                f"In a line from left to right: {', '.join(picks)}. "
                f"Who is {ordinal} from the left? Reply with just the name.",
                picks[target_pos], "spatial",
            ))
        elif kind == 2:
            # Stack of books
            subjects = rng.sample(
                ["Math", "Science", "History", "Art", "Biology", "Chemistry",
                 "Physics", "Literature", "Music"], 5)
            pos = rng.choice([1, 2, 3])  # 0-indexed; position 2, 3, 4 from bottom
            ordinal = {1: "second", 2: "third", 3: "fourth"}[pos]
            out.append((
                f"Five books are stacked from bottom to top: {', '.join(subjects)}. "
                f"Which book is {ordinal} from the bottom? Reply with just the subject name.",
                subjects[pos], "spatial",
            ))
        else:
            # Chain of left/right
            n = rng.sample(names, 4)
            out.append((
                f"{n[0]} is to the left of {n[1]}. {n[1]} is to the left of {n[2]}. "
                f"{n[2]} is to the left of {n[3]}. Who is on the far right? "
                "Reply with just the name.",
                n[3], "spatial",
            ))
    return out


def _gen_counting(rng: random.Random, k: int) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for i in range(k):
        kind = i % 5
        if kind == 0:
            # Handshakes
            n = rng.randint(3, 10)
            out.append((
                f"There are {n} people in a room. Each shakes hands with every "
                "other person exactly once. How many handshakes total?",
                str(n * (n - 1) // 2), "counting",
            ))
        elif kind == 1:
            # Red + blue marbles
            r = rng.randint(2, 12)
            b = rng.randint(2, 12)
            out.append((
                f"A bag has {r} red marbles and {b} blue marbles. How many marbles total?",
                str(r + b), "counting",
            ))
        elif kind == 2:
            # Days in N weeks
            n = rng.randint(2, 8)
            out.append((
                f"How many days are in {n} weeks?",
                str(n * 7), "counting",
            ))
        elif kind == 3:
            # Pizza slices
            total = rng.choice([8, 10, 12])
            eaten = rng.randint(2, total - 2)
            out.append((
                f"A pizza is cut into {total} slices. {eaten} slices are eaten. "
                "How many slices are left?",
                str(total - eaten), "counting",
            ))
        else:
            # Geometric shape corners
            shape, corners = rng.choice([
                ("cube", "8"), ("square", "4"), ("triangle", "3"),
                ("pentagon", "5"), ("hexagon", "6"), ("octagon", "8"),
            ])
            out.append((
                f"How many corners (vertices) does a {shape} have?",
                corners, "counting",
            ))
    return out


def _gen_sequence(rng: random.Random, k: int) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for i in range(k):
        kind = i % 4
        if kind == 0:
            # Arithmetic
            start = rng.randint(1, 12)
            step = rng.randint(2, 7)
            seq = [start + j * step for j in range(5)]
            nxt = start + 5 * step
            out.append((
                f"What comes next in this sequence: {', '.join(map(str, seq))}, ?",
                str(nxt), "sequence",
            ))
        elif kind == 1:
            # Geometric
            start = rng.choice([1, 2, 3])
            ratio = rng.choice([2, 3])
            seq = [start * ratio ** j for j in range(5)]
            nxt = start * ratio ** 5
            out.append((
                f"What comes next: {', '.join(map(str, seq))}, ?",
                str(nxt), "sequence",
            ))
        elif kind == 2:
            # Squares starting from some point
            start = rng.randint(1, 4)
            seq = [(start + j) ** 2 for j in range(4)]
            nxt = (start + 4) ** 2
            out.append((
                f"What comes next in this sequence of perfect squares: "
                f"{', '.join(map(str, seq))}, ?",
                str(nxt), "sequence",
            ))
        else:
            # Missing in arithmetic
            start = rng.randint(1, 8)
            step = rng.randint(2, 6)
            seq = [start + j * step for j in range(6)]
            missing_pos = rng.choice([2, 3])
            missing = seq[missing_pos]
            display = [str(x) if j != missing_pos else "?" for j, x in enumerate(seq)]
            out.append((
                f"What number is missing: {', '.join(display)}",
                str(missing), "sequence",
            ))
    return out


def _gen_conditional(rng: random.Random, k: int) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    scenarios = [
        ("it rains", "the streets get wet", "no", "no"),       # modus tollens
        ("you study", "you pass the test", "yes", "yes"),       # modus ponens
        ("a number is divisible by 4", "it is divisible by 2", "no", None),  # special
        ("a shape is a triangle", "it has 3 sides", "no", None),  # special
    ]
    for i in range(k):
        kind = i % 4
        if kind == 0:
            # Modus tollens: If P then Q. Not Q. Did P happen?
            p, q, ans_no, _ = scenarios[0]
            out.append((
                f"If {p}, {q}. {q.capitalize()} did not happen. Did {p}? "
                "Reply with only 'yes' or 'no'.",
                ans_no, "conditional",
            ))
        elif kind == 1:
            # Modus ponens
            p, q, ans_yes, _ = scenarios[1]
            actor = rng.choice(["Mary", "Tom", "Lisa", "John", "Anna"])
            out.append((
                f"If {p}, {q}. {actor} did study. Did {actor} pass the test? "
                "Assume the premise is true. Reply with only 'yes' or 'no'.",
                ans_yes, "conditional",
            ))
        elif kind == 2:
            # Specific: divisibility by 4 → 2
            n = rng.choice([6, 10, 14, 18, 22])
            # All these are divisible by 2 but not 4
            out.append((
                f"If a number is divisible by 4, it is divisible by 2. "
                f"The number {n} is divisible by 2. Is {n} divisible by 4? "
                "Reply with only 'yes' or 'no'.",
                "no", "conditional",
            ))
        else:
            # Affirming antecedent that leads to no
            day_today = rng.choice(["Wednesday", "Friday", "Saturday", "Sunday"])
            out.append((
                f"If today is Monday, tomorrow is Tuesday. Today is {day_today}. "
                "Is tomorrow Tuesday? Reply with only 'yes' or 'no'.",
                "no", "conditional",
            ))
    return out


def _gen_syllogism(rng: random.Random, k: int) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for i in range(k):
        kind = i % 4
        if kind == 0:
            # Modus ponens style: All A are B. X is A. Is X B? -> yes
            categories = rng.choice([
                ("mammals", "warm-blooded", "Whales", "mammals"),
                ("birds", "egg-layers", "Sparrows", "birds"),
                ("cats", "carnivores", "Tigers", "cats"),
                ("flowers", "plants", "Roses", "flowers"),
                ("squares", "rectangles", "This shape", "a square"),
            ])
            cat, prop, x, x_kind = categories
            out.append((
                f"All {cat} are {prop}. {x} are {x_kind}. Are {x} {prop}? "
                "Reply with only 'yes' or 'no'.",
                "yes", "syllogism",
            ))
        elif kind == 1:
            # Transitive: All A are B. All B are C. Are all A C? -> yes
            chain = rng.choice([
                ("cats", "mammals", "animals"),
                ("squares", "rectangles", "quadrilaterals"),
                ("dogs", "canines", "mammals"),
                ("oaks", "trees", "plants"),
                ("trout", "fish", "vertebrates"),
            ])
            a, b, c = chain
            out.append((
                f"All {a} are {b}. All {b} are {c}. Are all {a} {c}? "
                "Reply with only 'yes' or 'no'.",
                "yes", "syllogism",
            ))
        elif kind == 2:
            # Invalid: All A are B. Some B are C. Therefore some A are C.
            # This is the "undistributed middle" fallacy. Answer: no.
            chain = rng.choice([
                ("birds", "egg-layers", "reptiles"),
                ("dogs", "mammals", "elephants"),
                ("roses", "red things", "stop signs"),
                ("squares", "shapes", "circles"),
            ])
            a, b, c = chain
            out.append((
                f"All {a} are {b}. Some {b} are {c}. "
                f"Therefore some {a} are {c}. Is this argument logically valid? "
                "Reply with only 'yes' or 'no'.",
                "no", "syllogism",
            ))
        else:
            # Valid: All A are B. No B are C. So no A are C. -> yes
            triplet = rng.choice([
                ("apples", "fruits", "vegetables"),
                ("dogs", "mammals", "reptiles"),
                ("squares", "shapes", "sounds"),
                ("roses", "flowers", "rocks"),
            ])
            a, b, c = triplet
            out.append((
                f"All {a} are {b}. No {b} are {c}. "
                f"Therefore no {a} are {c}. Is this argument logically valid? "
                "Reply with only 'yes' or 'no'.",
                "yes", "syllogism",
            ))
    return out


# ============================================================================
# Eval mechanics — same prompts as before
# ============================================================================


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
    return re.sub(r"[^\w]", "", str(s).lower())


def extract_answer(text: str) -> str:
    m = re.search(r"answer\s*[:=]\s*([^\n]+)", text, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().rstrip(".,;:")
        first = candidate.split()[0] if candidate else ""
        return _normalize(first)
    words = text.strip().split()
    return _normalize(words[-1]) if words else ""


def is_correct(extracted: str, expected: str) -> bool:
    return extracted == _normalize(expected)


@dataclass
class ArmResult:
    scores: np.ndarray
    responses: list[str]
    guesses: list[str]


def run_arm(
    template: str, label: str, problems: list[tuple[str, str, str]]
) -> ArmResult:
    print(f"\n--- {label} ---", flush=True)
    scores = np.zeros(len(problems), dtype=np.float64)
    responses: list[str] = []
    guesses: list[str] = []
    t0 = time.time()
    for i, (q, correct, _cat) in enumerate(problems):
        response = ask_phi3(template.format(q=q))
        guess = extract_answer(response)
        score = 1.0 if is_correct(guess, correct) else 0.0
        scores[i] = score
        responses.append(response)
        guesses.append(guess)
        # Progress dots — print every 10 problems
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            running_acc = scores[: i + 1].mean()
            print(
                f"  [{i + 1}/{len(problems)}] running accuracy: "
                f"{running_acc:.0%}  elapsed: {elapsed:.0f}s "
                f"(~{elapsed / (i + 1):.1f}s/problem)",
                flush=True,
            )
    print(
        f"  FINAL: {scores.mean():.0%}  ({int(scores.sum())}/{len(scores)})  "
        f"elapsed: {time.time() - t0:.0f}s",
        flush=True,
    )
    return ArmResult(scores, responses, guesses)


def main() -> int:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: Ollama not reachable ({e})", file=sys.stderr)
        return 1

    problems = generate_problems(seed=42)
    print(
        f"Caliber n=100 two-prompt comparison — {MODEL} via Ollama", flush=True
    )
    print(f"Task: {len(problems)} generated logic/reasoning problems")
    print(
        f"Categories: "
        f"{', '.join(sorted({p[2] for p in problems}))}"
    )
    print(f"Prompt 1: Freeform 'think step by step'")
    print(f"Prompt 2: Structured 'GIVEN / FIND / SOLVE'")
    print(f"Pipeline: compare -> explain -> judge_hypothesis")
    print(f"Estimated runtime: ~20 minutes (200 phi3 calls)")

    arm1 = run_arm(PROMPT_1_FREEFORM,   "Prompt 1: Freeform CoT", problems)
    arm2 = run_arm(PROMPT_2_STRUCTURED, "Prompt 2: Structured CoT", problems)

    # =====================================================================
    # Layer 1 — verdict
    # =====================================================================
    print()
    print("=" * 72)
    print("LAYER 1 — caliber.compare()  (which prompt is statistically better?)")
    print("=" * 72)
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

    if verdict.verdict == "BETTER":
        winner = "Prompt 2 (Structured GIVEN/FIND/SOLVE)"
        loser = "Prompt 1 (Freeform)"
    elif verdict.verdict == "WORSE":
        winner = "Prompt 1 (Freeform)"
        loser = "Prompt 2 (Structured GIVEN/FIND/SOLVE)"
    elif verdict.verdict == "NO_CHANGE":
        winner = None
        loser = None
        print("Verdict: NO_CHANGE")
    else:
        winner = None
        loser = None
        print(f"Verdict: INCONCLUSIVE at n={len(arm1.scores)} — these prompts are statistically equivalent.")

    if winner:
        print(f">>> WINNER:  {winner}")
        print(f">>> LOSER:   {loser}")

    # =====================================================================
    # Layer 2 — explain
    # =====================================================================
    print()
    print("=" * 72)
    print("LAYER 2 — caliber.explain()  (where did the difference come from?)")
    print("=" * 72)
    explanation = caliber.explain(
        arm1.scores, arm2.scores,
        inputs=[p[0] for p in problems],
        old_outputs=arm1.responses,
        new_outputs=arm2.responses,
        categories=[p[2] for p in problems],
        top_n_examples=3,
        seed=0,
    )

    print()
    print("Stratified breakdown:")
    print(f"  {'category':<12}  {'n':>3}  {'P1':>6}  {'P2':>6}  {'delta':>7}")
    print("  " + "-" * 44)
    for s in sorted(explanation.strata, key=lambda x: -x.delta):
        print(f"  {s.name:<12}  {s.n:>3}  {s.old_accuracy:>5.0%}  {s.new_accuracy:>5.0%}  {s.delta:>+6.0%}")
    if explanation.biggest_gain_category and explanation.smallest_gain_category and \
            explanation.biggest_gain_category != explanation.smallest_gain_category:
        print()
        print(f"  Where Prompt 2 helped most:   {explanation.biggest_gain_category}")
        print(f"  Where Prompt 2 helped least:  {explanation.smallest_gain_category}")

    if explanation.top_improvements:
        print()
        print(f"Top {len(explanation.top_improvements)} cases where Prompt 2 beat Prompt 1:")
        for ex in explanation.top_improvements:
            cat = f"({ex.category})" if ex.category else ""
            q_short = ex.input if ex.input and len(ex.input) <= 60 else (ex.input[:57] + "..." if ex.input else "")
            print(f"  [#{ex.index:2d}] {cat:<12}  {q_short}")
    if explanation.top_regressions:
        print()
        print(f"Top {len(explanation.top_regressions)} cases where Prompt 1 beat Prompt 2:")
        for ex in explanation.top_regressions:
            cat = f"({ex.category})" if ex.category else ""
            q_short = ex.input if ex.input and len(ex.input) <= 60 else (ex.input[:57] + "..." if ex.input else "")
            print(f"  [#{ex.index:2d}] {cat:<12}  {q_short}")

    # =====================================================================
    # Layer 3 — judge_hypothesis
    # =====================================================================
    print()
    print("=" * 72)
    print("LAYER 3 — caliber.judge_hypothesis()  (why is the winner better?)")
    print("=" * 72)

    if verdict.verdict == "BETTER" and len(explanation.top_improvements) >= 2:
        provider = caliber.OllamaProvider(model=MODEL, num_predict=400)
        print(f"\nAsking {provider.name}/{provider.model} for the pattern...", flush=True)
        t0 = time.time()
        judged = caliber.judge_hypothesis(explanation, provider, n_examples=3)
        print(f"  elapsed: {time.time() - t0:.1f}s")
        print()
        print("Hypothesis — Prompt 2 was better because:")
        print("-" * 72)
        print(judged.hypothesis)
        print("-" * 72)
    elif verdict.verdict == "WORSE" and len(explanation.top_regressions) >= 2:
        inverted = caliber.explain(
            arm2.scores, arm1.scores,
            inputs=[p[0] for p in problems],
            old_outputs=arm2.responses,
            new_outputs=arm1.responses,
            categories=[p[2] for p in problems],
            top_n_examples=3,
            seed=0,
        )
        provider = caliber.OllamaProvider(model=MODEL, num_predict=400)
        print(f"\nAsking {provider.name}/{provider.model} for the pattern...", flush=True)
        t0 = time.time()
        judged = caliber.judge_hypothesis(inverted, provider, n_examples=3)
        print(f"  elapsed: {time.time() - t0:.1f}s")
        print()
        print("Hypothesis — Prompt 1 was better because:")
        print("-" * 72)
        print(judged.hypothesis)
        print("-" * 72)
    else:
        print(f"\nSkipping judge — verdict is {verdict.verdict}, no clear winner.")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
