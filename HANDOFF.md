# Caliber — Session Handoff

> **Purpose:** This file lets a new chat session (Claude or human) pick up
> exactly where the last one ended. Update it at the end of each working
> day with where we are, what's next, and what's blocking.

---

## What Caliber is

A Python library that gives AI engineers a statistically rigorous verdict
on whether a prompt/model/agent change is **real or just noise.** Beyond the
verdict, it explains **where the gain came from** (stratification) and
**why** (LLM-judged hypothesis). Apache 2.0, public:
https://github.com/karyampudilikhit/caliber-eval

The goal: turn this from a library into a SaaS with **100 paying customers
in 6 months.** User is solo founder; treats Claude as co-founder.

---

## Quick-start for a new session

```bash
# Look at code & tests
cd D:/quant/caliber-eval
git log --oneline -10
python -m pytest tests/ -q              # should be all green
gh repo view --json url,description     # confirm public repo

# Resume the daily co-founder rhythm
```

When you (Claude) start a fresh chat, the user will likely say something like
"continuing Caliber" or "day 3 plan". Read this file + the three Caliber
memories first, then give a morning brief.

---

## State of the company

**As of end of Day 1 (2026-06-23):**

| | |
|---|---|
| Product | v1 library complete: `compare`, `explain`, `judge_hypothesis`, `sample_size`, `benjamini_hochberg`, `SequentialTester`, `PageHinkleyDetector`, `CUSUMDetector`, CLI. 201 tests, 98% coverage. |
| Code | `D:/quant/caliber-eval` |
| Repo | https://github.com/karyampudilikhit/caliber-eval (public, Apache 2.0) |
| CI | Green on Python 3.10/3.11/3.12 × ubuntu/macos/windows |
| PyPI | ❌ NOT YET PUBLISHED. `pip install caliber-eval` does not work. Top priority for Day 2. |
| Customers / users / waitlist | 0 / 0 / 0 |
| Marketing assets shipped | None public yet — MLOps Slack post drafted but auto-flagged on first attempt and needs reposting cleaned. LinkedIn post drafted but not posted. |
| Hero example | n=100 test ran end-of-Day-1: verdict INCONCLUSIVE at -3pp (CI -13%, +7%) across 5 categories of logic problems. Demonstrates Caliber correctly refusing to call winners that aren't there. Use this in launch content. |

---

## What got done on Day 1

- ✅ Built full Caliber v1: `compare`, `explain`, `judge_hypothesis` (Tier 1-3 of the "why" architecture)
- ✅ Made repo public on GitHub
- ✅ CI green across 9 platform/version combos
- ✅ Updated README with the full `explain()` + `judge_hypothesis()` examples
- ✅ Ran three real-LLM tests against local phi3 (multiplication, complex math, word problems)
- ✅ Ran n=10 / n=30 / n=100 two-prompt comparison — n=100 confirmed the prompts are statistically equivalent (great hero finding)
- ✅ Saved 3 memory files for cross-session context
- ✅ Wrote this handoff doc

## What did NOT get done on Day 1

- ❌ PyPI not published (waiting on user's PyPI account + API token)
- ❌ Cleaned MLOps Slack post not yet reposted (waiting on user)
- ❌ Cold outreach targets not identified (waiting on user)
- ❌ Landing page + Tally waitlist not built
- ❌ LinkedIn post not posted

These all rolled to Day 2.

---

## Day 2 plan (tomorrow)

### What only the founder can do

1. **PyPI account creation** — https://pypi.org/account/register/, then https://pypi.org/manage/account/token/ → API token, paste it to chat. (~10 min)
2. **Repost cleaned MLOps Slack message in MLOps Community Slack `#showcase`.** Use the cleaned version (no "shit", no "take it on the chin"). Reference the n=100 result as evidence. (~5 min)
3. **Identify 10 cold-outreach targets.** Named humans, not job titles. From GitHub stargazers of `braintrustdata/braintrust-sdk`, `langfuse/langfuse`, `langchain-ai/langsmith`, or recent posters in MLOps Community Slack `#evals`. For each: name + LinkedIn URL + one sentence about why they signal "cares about eval rigor." (~45 min)
4. **Tell Claude how many hours/week you can really commit.** Calibrates the 6-month plan honestly.

### What Claude does in parallel

- **Push to PyPI** (the moment you give me the token). I'll run `python -m build && twine upload dist/*`. Then `pip install caliber-eval` works for everyone.
- **Draft personalized cold DMs** for each of the 10 targets you identify. One paragraph each. You send.
- **Build a one-page landing site** — single HTML file with headline, one paragraph, Tally waitlist form embed, GitHub link. You deploy to Vercel in 10 min.
- **Sharpen the LinkedIn post** with the n=100 result as a hero example: *"I tested two prompt variants on 100 problems. The naïve view at n=10 said Prompt 1 wins by 20 points. At n=100 it was 3 points. Here's the tool that wouldn't let me ship the false finding."*
- **Write the n=100 result as a tweet thread** (5-7 tweets) for X/Twitter.

### Day 2 success criteria

- ✅ `pip install caliber-eval` works
- ✅ MLOps Slack post is live, no profanity flag
- ✅ 10 cold outreach targets identified
- ✅ At least 3 cold DMs sent

If all 4 land, Day 2 is a win.

---

## Open decisions the founder needs to make

These are blocking and only the founder can decide:

1. **Hours per week** — recalibrates the whole plan
2. **Pricing thesis** — Claude proposed free OSS + $99/mo single + $499/mo team. Confirm or counter.
3. **Paying vs free customers** — when we say "100 customers," paying or free trial? Affects strategy materially.
4. **Cloud LLM vs local-only** — should `OllamaProvider` be joined by `OpenAIProvider` / `AnthropicProvider` in the library, or does the user implement those on their own using our protocol? Default: ship our own to lower friction; user pays the API cost. Confirm.

---

## Files of note in the repo

- `README.md` — public-facing pitch and quickstart
- `caliber/` — the library source
- `tests/` — 201 tests, 98% coverage
- `examples/01_decision.ipynb`, `examples/02_monitoring.ipynb` — runnable notebooks
- `real_eval_test.py`, `real_eval_test_complex.py`, `real_eval_test_words.py`,
  `real_eval_full_pipeline.py`, `real_eval_two_prompts.py`,
  `real_eval_two_prompts_v2.py`, `real_eval_n100.py` —
  end-to-end LLM tests against local phi3. The last one (`real_eval_n100.py`)
  produces the hero result for marketing.
- `n100_output.log` — captured output from the n=100 run (use for content)
- `HANDOFF.md` — this file

---

## How the daily rhythm works

**Every morning** the user pings Claude → Claude reads this file + memories → gives a brief:
1. State of the company (delta from yesterday)
2. Today's 3-5 tasks (ranked by leverage)
3. What only the founder can do
4. What Claude does in parallel
5. End-of-day check-in

**Every Friday**: weekly review — what we did, what we didn't, what we learned, recalibrate.

**Update this file** at the end of each day before context fills up. The user
or Claude should rewrite the "Day N plan" section, push the previous day's plan
down into a "Past days" section if needed, and mark progress on open decisions.
