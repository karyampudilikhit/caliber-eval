# Caliber

**Is this LLM change actually better, or just noise?**

You ran evals on two versions of a prompt, model, or agent. The mean score moved. Caliber tells you whether the change is *statistically real* — with a confidence interval, a p-value, and a recommended next step.

```python
import caliber

result = caliber.compare(old_judge_scores, new_judge_scores)

print(result.verdict)         # BETTER | WORSE | INCONCLUSIVE | NO_CHANGE
print(result.ci)              # (lower, upper) — 95% CI on the mean difference
print(result.recommendation)  # human-readable next step
```

Caliber plugs into your existing eval stack. It doesn't replace it — it adds statistical rigor on top.

---

## Install

```bash
pip install caliber-eval
```

Requires Python 3.10+. Runs on Linux, macOS, Windows.

---

## Quickstart

```python
import numpy as np
import caliber

# Eval scores from two prompt versions, scored by the same judge on the same dataset.
# In real life these come from your eval runner (Braintrust, Langfuse, LangSmith, custom).
rng = np.random.default_rng(0)
old = rng.normal(0.60, 0.05, 100)   # baseline prompt
new = rng.normal(0.65, 0.05, 100)   # new prompt — same eval set

result = caliber.compare(old, new, metric_name="accuracy")
print(result.recommendation)
```

```
Ship the new version. accuracy improved by +0.0498
(95% CI: [+0.0359, +0.0637], n=100). The interval excludes zero.
```

That's the core loop. Two arrays in, a verdict and a plain-English recommendation out.

---

## The four verdicts

| Verdict | What it means | What to do |
|---|---|---|
| **BETTER** | The improvement is statistically significant — the CI excludes zero | Ship it |
| **WORSE** | The regression is statistically significant | Don't ship |
| **NO_CHANGE** | `|mean_difference|` is below your practical threshold | Save your effort |
| **INCONCLUSIVE** | Could be a real effect but too noisy at this n to call | Collect more samples |

Use `practical_threshold=0.05` (or whatever your "worth shipping for" floor is) so trivially-significant-but-tiny effects are correctly labeled `NO_CHANGE`.

---

## More workflows

### "How many samples do I need?"

```python
# I care about detecting a 5-accuracy-point improvement with 80% power,
# given baseline judge-score noise of σ=0.10.
r = caliber.sample_size(
    effect_size=0.05,
    effect_type="absolute",
    baseline_std=0.10,
    power=0.8,
)
print(r.n_per_arm)  # 34
```

### "I'm comparing many metrics at once — how do I correct?"

```python
results = [caliber.compare(baseline, variant) for variant in metric_arms]
p_values = [r.p_value for r in results]

# Benjamini-Hochberg FDR control at α=0.05
significant = caliber.benjamini_hochberg(p_values, alpha=0.05)
# significant[i] == True → variant i is a real winner after correction
```

### "Can I peek at results across batches?"

```python
# Plan for up to 5 looks at up to 500 total pairs. Stop as soon as the
# verdict is decisive — without inflating family-wise α.
tester = caliber.SequentialTester(max_n=500, n_looks=5, alpha=0.05)

for batch_old, batch_new in stream_of_batches():
    result = tester.update(batch_old, batch_new)
    print(f"look n={result.n}: {result.verdict}")
    if tester.is_done():
        break
```

The boundary tightens at each look using O'Brien-Fleming alpha-spending. The naïve "peek and call it significant if p < 0.05" pattern is broken; this is the fix.

### "Is my production score series drifting?"

```python
detector = caliber.PageHinkleyDetector(delta=0.05, threshold=50)

for score in production_score_stream():
    event = detector.add(score)
    if event is not None:
        alert(
            f"Drift at index {event.detected_at_index}: "
            f"mean {event.mean_before:.3f} -> {event.mean_after:.3f}"
        )
        detector.reset()
```

Or use `CUSUMDetector(target_mean=..., target_std=...)` when you know the H₀ target.

### "OK, the change is real — but WHERE did the gain come from?"

`compare()` tells you whether the difference is statistically real. `explain()`
breaks it down by category and surfaces the specific examples that drove it:

```python
r = caliber.explain(
    old_scores, new_scores,
    inputs=questions,                # the prompts/questions (optional)
    old_outputs=old_model_responses, # what the old model said (optional)
    new_outputs=new_model_responses, # what the new model said (optional)
    categories=["math", "math", "word", "code", ...],  # one tag per example
    top_n_examples=3,
)

print(r.summary)
```

```
Verdict: BETTER (score Δ=+0.5333, 95% CI [+0.37, +0.70], p=0.0001, n=30)

By category (sorted by delta):
  PEMDAS      n=5    old=0%    new=100%  Δ +100%
  algebra     n=5    old=0%    new=100%  Δ +100%
  word        n=10   old=40%   new=90%   Δ +50%
  percent     n=5    old=80%   new=100%  Δ +20%
  powers      n=5    old=100%  new=100%  Δ +0%

Largest gain: PEMDAS.  Smallest gain: powers.

Top 3 improvement(s):
  [#0] (PEMDAS)  Compute: 8 + 4 * 3 - 2     Δ +1.00
  [#3] (algebra) If 3x + 7 = 25, what is x? Δ +1.00
  [#6] (word)    The sum of three...        Δ +1.00
```

No LLM call — pure data analysis. The output tells you exactly which slice of
your eval the improvement came from, so you can write a sharper headline:
*"CoT helped on rule-application tasks but didn't help on simple lookups."*

---

## CLI

Every feature is also a shell command:

```bash
# Verdict on paired score CSVs
caliber compare old.csv new.csv --metric accuracy
caliber compare old.csv new.csv --json | jq .verdict

# Sample-size planning
caliber sample-size --effect 0.05 --effect-type absolute --baseline-std 0.1 --power 0.8

# Drift scan over a score stream
caliber drift scores.csv --threshold 5
caliber drift scores.csv --detector cusum --target-mean 0.5 --target-std 0.1 --h 5
```

CSV inputs auto-detect a header row. Every command takes `--json` for piping into `jq` or scripts.

If `caliber` isn't on your `PATH` after install, `python -m caliber.cli <command>` always works.

---

## Why statistical rigor

Caliber is the right tool when:

- Your eval set is **small** (n=20–100 is typical for LLM evals)
- Your scores are **noisy** (LLM-as-judge variance, prompt sensitivity, sampling temperature)
- A wrong verdict has **real cost** — shipping a regression, or missing a real win

Under the hood:

- **Paired methods by default.** Two prompts scored on the same eval set is more powerful as a paired comparison.
- **Auto-selection between paired-t and paired-bootstrap.** Paired-t when n ≥ 30 and the differences are roughly normal; bootstrap (with finite-sample corrections) otherwise.
- **Benjamini-Hochberg FDR control** when comparing many metrics.
- **Group-sequential testing** with O'Brien-Fleming or Pocock boundaries — peeking is safe.
- **Page-Hinkley and CUSUM drift detection** with documented false-positive properties.
- **Math validated by simulation.** Every shipped method has property-based tests verifying CI coverage ≥ 94%, false-positive rate ≤ α, and power-recovery within ±5%.

---

## What's in v1

| Module | Status |
|---|---|
| `caliber.compare` — verdict on paired scores | ✅ |
| `caliber.explain` — verdict + per-category breakdown + driving examples | ✅ |
| `caliber.sample_size` — required n for a target effect | ✅ |
| `caliber.benjamini_hochberg` — FDR correction | ✅ |
| `caliber.SequentialTester` — group-sequential design | ✅ |
| `caliber.PageHinkleyDetector` / `CUSUMDetector` — drift detection | ✅ |
| CLI: `caliber compare` / `sample-size` / `drift` | ✅ |
| Adapters for Braintrust / Langfuse / LangSmith | 🚧 v1.1 |

See [`examples/`](examples/) for runnable notebooks.

---

## Development

```bash
git clone https://github.com/karyampudilikhit/caliber-eval.git
cd caliber-eval
pip install -e ".[dev]"

ruff check caliber/ tests/
mypy caliber/
pytest tests/                              # 150+ tests, ~15s
pytest tests/ --cov=caliber                # with coverage
```

---

## License

Apache 2.0.
