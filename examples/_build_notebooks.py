"""Builder for the example notebooks.

Run this from the project root to regenerate `examples/01_decision.ipynb`
and `examples/02_monitoring.ipynb`. The notebooks themselves are version-
controlled — this script just produces them so the content is reviewable
as plain Python rather than 1000-line JSON diffs.
"""

from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ============================================================================
# 01_decision.ipynb — the verdict workflow
# ============================================================================

DECISION_CELLS = [
    md(
        "# Caliber — the decision workflow\n"
        "\n"
        "You changed a prompt. Eval scores moved. Should you ship?\n"
        "\n"
        "This notebook walks the full decision workflow on a simulated LLM eval:\n"
        "\n"
        "1. `caliber.compare` — verdict + CI on paired scores\n"
        "2. The four verdicts in action\n"
        "3. `caliber.sample_size` — plan before you run\n"
        "4. `caliber.benjamini_hochberg` — many metrics at once\n"
        "5. `caliber.SequentialTester` — peek across batches"
    ),
    code(
        "import numpy as np\n"
        "import caliber\n"
        "\n"
        "print('caliber', caliber.__version__)"
    ),
    md(
        "## 1. The basic verdict\n"
        "\n"
        "Two prompt versions, scored by the same judge on the same 100 examples.\n"
        "The new prompt is genuinely a little better (mean 0.65 vs 0.60)."
    ),
    code(
        "rng = np.random.default_rng(0)\n"
        "old = rng.normal(0.60, 0.05, 100)\n"
        "new = rng.normal(0.65, 0.05, 100)\n"
        "\n"
        "result = caliber.compare(old, new, metric_name='accuracy')\n"
        "print(f'verdict:           {result.verdict}')\n"
        "print(f'mean difference:   {result.mean_difference:+.4f}')\n"
        "print(f'95% CI:            {result.ci}')\n"
        "print(f'p-value:           {result.p_value:.4g}')\n"
        "print(f'method:            {result.method}')\n"
        "print()\n"
        "print(result.recommendation)"
    ),
    md(
        "## 2. The four verdicts\n"
        "\n"
        "Simulate one scenario for each verdict the library can return."
    ),
    code(
        "rng = np.random.default_rng(1)\n"
        "\n"
        "# BETTER — clear improvement.\n"
        "old = rng.normal(0.60, 0.05, 100); new = rng.normal(0.75, 0.05, 100)\n"
        "print('BETTER       ->', caliber.compare(old, new).verdict)\n"
        "\n"
        "# WORSE — clear regression.\n"
        "old = rng.normal(0.75, 0.05, 100); new = rng.normal(0.60, 0.05, 100)\n"
        "print('WORSE        ->', caliber.compare(old, new).verdict)\n"
        "\n"
        "# INCONCLUSIVE — too noisy at this n to call.\n"
        "old = rng.normal(0.70, 0.20, 20); new = rng.normal(0.72, 0.20, 20)\n"
        "print('INCONCLUSIVE ->', caliber.compare(old, new, seed=0).verdict)\n"
        "\n"
        "# NO_CHANGE — real effect but below the practical threshold.\n"
        "old = rng.normal(0.700, 0.005, 200); new = rng.normal(0.703, 0.005, 200)\n"
        "print('NO_CHANGE    ->', caliber.compare(old, new, practical_threshold=0.05).verdict)"
    ),
    md(
        "## 3. Auto-selection between paired-t and bootstrap\n"
        "\n"
        "With `method='auto'` (the default), Caliber picks paired-t when n ≥ 30 and the\n"
        "differences look roughly normal; otherwise it falls back to the bootstrap. You\n"
        "don't have to think about it."
    ),
    code(
        "small = caliber.compare(\n"
        "    rng.normal(0.5, 0.1, 20),\n"
        "    rng.normal(0.55, 0.1, 20),\n"
        "    seed=0,\n"
        ")\n"
        "print(f'n=20  -> method={small.method}')\n"
        "\n"
        "large = caliber.compare(\n"
        "    rng.normal(0.5, 0.1, 100),\n"
        "    rng.normal(0.55, 0.1, 100),\n"
        ")\n"
        "print(f'n=100 -> method={large.method}')"
    ),
    md(
        "## 4. Plan before you run: `sample_size`\n"
        "\n"
        "You can't decide *after* the eval whether your sample was big enough — that's\n"
        "p-hacking. Decide *before*: what effect do you care about, and how confident\n"
        "do you want to be?"
    ),
    code(
        "# I care about a 5-accuracy-point improvement, baseline judge noise σ=0.10,\n"
        "# 80% power. How many paired evals do I need?\n"
        "needed = caliber.sample_size(\n"
        "    effect_size=0.05,\n"
        "    effect_type='absolute',\n"
        "    baseline_std=0.10,\n"
        "    power=0.8,\n"
        ")\n"
        "print(f'n = {needed.n_per_arm}')\n"
        "\n"
        "# Trade-offs — power vs sample size at the same effect:\n"
        "for power in (0.7, 0.8, 0.9, 0.95):\n"
        "    r = caliber.sample_size(\n"
        "        effect_size=0.05, effect_type='absolute', baseline_std=0.10, power=power,\n"
        "    )\n"
        "    print(f'  power={power:.2f} -> n={r.n_per_arm}')"
    ),
    md(
        "## 5. Many metrics at once: Benjamini-Hochberg\n"
        "\n"
        "If you compare 20 metrics and call anything with p < 0.05 a win, you'll see\n"
        "false positives. BH controls the false discovery rate."
    ),
    code(
        "# Simulate 20 metrics: only 4 actually moved; the other 16 are noise.\n"
        "rng = np.random.default_rng(7)\n"
        "n = 50\n"
        "true_winners = {0, 5, 10, 15}\n"
        "p_values = []\n"
        "for i in range(20):\n"
        "    delta = 0.05 if i in true_winners else 0.0\n"
        "    old = rng.normal(0.7, 0.1, n)\n"
        "    new = rng.normal(0.7 + delta, 0.1, n)\n"
        "    p_values.append(caliber.compare(old, new).p_value)\n"
        "\n"
        "rejected = caliber.benjamini_hochberg(p_values, alpha=0.05)\n"
        "naive    = [p < 0.05 for p in p_values]\n"
        "print(f'naive p<0.05 rejected: {sum(naive)} (incl. false positives)')\n"
        "print(f'BH-corrected:         {sum(rejected)} (FDR controlled)')\n"
        "print('true winners caught by BH:',\n"
        "      [i for i in true_winners if rejected[i]])"
    ),
    md(
        "## 6. Peeking safely: `SequentialTester`\n"
        "\n"
        "You're running evals batch by batch. You want to stop as soon as the answer\n"
        "is clear — but the naïve \"re-test at every batch\" pattern inflates the\n"
        "false-positive rate. Group-sequential testing fixes this with a boundary\n"
        "that tightens at each look."
    ),
    code(
        "tester = caliber.SequentialTester(max_n=500, n_looks=5, alpha=0.05)\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "for k in range(5):\n"
        "    batch_old = rng.normal(0.50, 0.10, 50)\n"
        "    batch_new = rng.normal(0.55, 0.10, 50)\n"
        "    result = tester.update(batch_old.tolist(), batch_new.tolist())\n"
        "    print(f'look {k+1}/5  n={result.n:3d}  verdict={result.verdict}')\n"
        "    if tester.is_done():\n"
        "        print()\n"
        "        print(result.recommendation)\n"
        "        break"
    ),
    md(
        "## Summary\n"
        "\n"
        "Every workflow above is a single call. Caliber is intentionally a small\n"
        "library: it solves one problem (is this change real?) with statistically\n"
        "rigorous defaults, and gets out of the way."
    ),
]


# ============================================================================
# 02_monitoring.ipynb — drift detection on a production stream
# ============================================================================

MONITORING_CELLS = [
    md(
        "# Caliber — production monitoring\n"
        "\n"
        "Your eval looks fine in dev. You ship it. A few weeks later production\n"
        "starts behaving differently — model upgraded under you, user mix shifted,\n"
        "a tool dependency changed. You want to catch this without staring at a\n"
        "dashboard.\n"
        "\n"
        "Caliber ships two streaming change-point detectors:\n"
        "\n"
        "- **Page-Hinkley** — learns the baseline from the data; use when you\n"
        "  don't know the target mean in advance.\n"
        "- **CUSUM** — tighter and more sensitive when you *do* know the\n"
        "  target mean and noise scale."
    ),
    code(
        "import numpy as np\n"
        "import caliber\n"
        "\n"
        "print('caliber', caliber.__version__)"
    ),
    md(
        "## Simulate a production score stream\n"
        "\n"
        "600 samples. The first 300 are at mean 0.70 (stable). At sample 300 the\n"
        "score drops to 0.55 — something changed."
    ),
    code(
        "rng = np.random.default_rng(0)\n"
        "stationary = rng.normal(0.70, 0.10, 300)\n"
        "drifted    = rng.normal(0.55, 0.10, 300)\n"
        "stream     = np.concatenate([stationary, drifted])\n"
        "print('stream length:', len(stream))\n"
        "print(f'mean before idx 300: {stationary.mean():.3f}')\n"
        "print(f'mean after  idx 300: {drifted.mean():.3f}')"
    ),
    md(
        "## Page-Hinkley\n"
        "\n"
        "Default parameters `delta=0.05, threshold=50` are very conservative — good\n"
        "for low false-alarm rate. For faster detection use a smaller threshold\n"
        "tuned to your noise scale; for fewer false alarms use a larger one."
    ),
    code(
        "detector = caliber.PageHinkleyDetector(delta=0.05, threshold=5.0)\n"
        "\n"
        "for i, score in enumerate(stream):\n"
        "    event = detector.add(float(score))\n"
        "    if event is not None:\n"
        "        print(\n"
        "            f'fired at sample {i}: '\n"
        "            f'mean {event.mean_before:.3f} -> {event.mean_after:.3f}  '\n"
        "            f'(magnitude {event.magnitude:.3f}, '\n"
        "            f'post-hoc p={event.p_value:.3g})'\n"
        "        )\n"
        "        break  # one alarm is enough for the demo\n"
        "else:\n"
        "    print('no drift detected')"
    ),
    md(
        "## CUSUM — when the target is known\n"
        "\n"
        "If you can say \"the score should be 0.70 ± 0.10 under H₀\", CUSUM is the\n"
        "stronger detector. Same setup, different math.\n"
        "\n"
        "Note CUSUM doesn't know the shift is the new normal — if you don't\n"
        "reset and adopt the new baseline, it will keep alarming. In production\n"
        "you'd typically alert once, page someone, and decide whether to\n"
        "re-target."
    ),
    code(
        "detector = caliber.CUSUMDetector(\n"
        "    target_mean=0.70,\n"
        "    target_std=0.10,\n"
        "    k=0.5,  # half the shift size you want to catch quickly\n"
        "    h=8.0,  # decision threshold in σ units — higher = more conservative\n"
        ")\n"
        "\n"
        "for i, score in enumerate(stream):\n"
        "    event = detector.add(float(score))\n"
        "    if event is not None:\n"
        "        print(\n"
        "            f'fired at sample {i}: '\n"
        "            f'mean {event.mean_before:.3f} -> {event.mean_after:.3f}  '\n"
        "            f'(magnitude {event.magnitude:.3f}, '\n"
        "            f'post-hoc p={event.p_value:.3g})'\n"
        "        )\n"
        "        break  # one alarm is enough; CUSUM keeps firing until you retarget\n"
        "else:\n"
        "    print('no drift detected')"
    ),
    md(
        "## Choosing between them\n"
        "\n"
        "| Detector | When | Trade-off |\n"
        "|---|---|---|\n"
        "| `PageHinkleyDetector` | You don't know the target mean | Learns baseline; slower |\n"
        "| `CUSUMDetector` | You know the target mean + σ | Tighter detection; needs target |\n"
        "\n"
        "Both have well-studied false-positive properties — start with default\n"
        "parameters and tune `threshold` / `h` based on observed false-alarm rate."
    ),
    md(
        "## Production integration\n"
        "\n"
        "In a real service this lives in your score-logging path:\n"
        "\n"
        "```python\n"
        "detector = caliber.PageHinkleyDetector()  # at process start\n"
        "\n"
        "def on_eval_score(score: float) -> None:\n"
        "    event = detector.add(score)\n"
        "    if event is not None:\n"
        "        emit_alert(\n"
        "            'Eval score drift detected',\n"
        "            mean_before=event.mean_before,\n"
        "            mean_after=event.mean_after,\n"
        "            magnitude=event.magnitude,\n"
        "        )\n"
        "        detector.reset()\n"
        "```\n"
        "\n"
        "Or run it offline over a CSV from the shell:\n"
        "\n"
        "```bash\n"
        "caliber drift scores.csv --threshold 5\n"
        "```"
    ),
]


# ============================================================================
# Build & write
# ============================================================================


def main() -> None:
    here = Path(__file__).parent
    targets = [
        (here / "01_decision.ipynb", DECISION_CELLS),
        (here / "02_monitoring.ipynb", MONITORING_CELLS),
    ]
    for path, cells in targets:
        path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
        print(f"wrote {path.relative_to(here.parent)}")


if __name__ == "__main__":
    main()
