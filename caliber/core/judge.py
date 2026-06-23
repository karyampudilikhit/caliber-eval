"""LLM-judged hypothesis layer — Tier 3 of the explain architecture.

`compare()` answers "is the difference real?".
`explain()` answers "where did the gain come from?" + "which examples drove it?".
`judge_hypothesis()` answers "what's the pattern in those examples?" — by
asking an LLM to read them and articulate it in plain English.

Pluggable provider design
-------------------------
Any object with a ``chat(prompt: str) -> str`` method, a ``name``, and a
``model`` works as a provider. We ship one concrete implementation
(``OllamaProvider`` — runs locally, no API key) and a sketch for cloud
providers in the docstring.

The hypothesis is **AI-generated and unverified.** Caliber marks it as such
on the returned `JudgedHypothesis` so callers don't accidentally treat it
as a proven claim. Verifying *why* a prompt change works is a human task —
this layer just proposes a candidate explanation to start the review.

Examples
--------
>>> import caliber
>>> from caliber import OllamaProvider
>>>
>>> exp = caliber.explain(
...     old_scores=[0, 0, 0, 1, 1],
...     new_scores=[1, 1, 1, 1, 1],
...     inputs=["Q1", "Q2", "Q3", "Q4", "Q5"],
...     old_outputs=["wrong1", "wrong2", "wrong3", "ok4", "ok5"],
...     new_outputs=["right1", "right2", "right3", "ok4", "ok5"],
... )
>>> # Then (skipped because it requires a running Ollama):
>>> # j = caliber.judge_hypothesis(exp, OllamaProvider("phi3"))
>>> # print(j.hypothesis)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from caliber.core.types import (
    ExampleFlip,
    ExplanationResult,
    JudgedHypothesis,
)

_DEFAULT_OUTPUT_TRUNC = 600  # chars per LLM response shown to the judge
_MAX_PROMPT_EXAMPLES = 5     # cap to avoid blowing context windows


@runtime_checkable
class LLMProvider(Protocol):
    """Anything with ``chat()``, a ``name``, and a ``model`` is a provider.

    Implement this for any LLM backend. Example for OpenAI::

        class OpenAIProvider:
            name = "openai"
            def __init__(self, model: str = "gpt-4o-mini") -> None:
                from openai import OpenAI
                self.model = model
                self._client = OpenAI()

            def chat(self, prompt: str) -> str:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return resp.choices[0].message.content or ""
    """

    name: str
    model: str

    def chat(self, prompt: str) -> str:
        ...


class OllamaProvider:
    """Default judge provider — talks to a local Ollama server.

    Parameters
    ----------
    model : str, default "phi3"
        Ollama model tag. Whatever you have installed locally (``ollama list``).
    url : str, default "http://localhost:11434/api/generate"
        Ollama generate endpoint.
    temperature : float, default 0.0
        0.0 → deterministic; useful when you want reproducible hypotheses.
    num_predict : int, default 400
        Max tokens to generate. The judge prompt asks for 1-2 sentences;
        400 leaves plenty of room.
    timeout : float, default 180.0
        HTTP timeout in seconds — local LLMs can be slow on small machines.
    """

    name: str = "ollama"

    def __init__(
        self,
        model: str = "phi3",
        url: str = "http://localhost:11434/api/generate",
        temperature: float = 0.0,
        num_predict: int = 400,
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.url = url
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout

    def chat(self, prompt: str) -> str:
        # Imported lazily so users without `requests` don't pay an import cost.
        import requests

        r = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.num_predict,
                },
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return str(r.json()["response"])


def judge_hypothesis(
    explanation: ExplanationResult,
    provider: LLMProvider,
    *,
    n_examples: int = 3,
    require_at_least: int = 2,
    require_inputs_and_outputs: bool = True,
) -> JudgedHypothesis:
    """Have an LLM read the driving examples and propose a hypothesis for why
    the new prompt differs from the old.

    Parameters
    ----------
    explanation : ExplanationResult
        Output of ``caliber.explain()``. Its ``top_improvements`` are the
        cases shown to the judge.
    provider : LLMProvider
        Any object implementing the LLMProvider protocol — see
        ``OllamaProvider`` for the canonical concrete implementation.
    n_examples : int, default 3
        How many driving examples to include in the judge prompt. Capped at 5
        to keep prompts inside small-model context windows.
    require_at_least : int, default 2
        Minimum number of available driving examples to attempt judgement.
        Below this, raises ``ValueError`` — one example doesn't establish a
        pattern.
    require_inputs_and_outputs : bool, default True
        Require that the explanation's driving examples include both the
        original inputs and both model outputs. Without these, the LLM has
        nothing to reason about. Set ``False`` only for testing.

    Returns
    -------
    JudgedHypothesis
        The LLM's proposed pattern, marked as AI-generated.

    Raises
    ------
    ValueError
        If there are fewer than ``require_at_least`` driving examples, or
        ``require_inputs_and_outputs=True`` and inputs/outputs are missing.

    Notes
    -----
    The returned hypothesis is **not verified.** Caliber doesn't check whether
    the LLM's claim is correct — that's the user's job. Use this to seed
    review, not to replace it.
    """
    available = list(explanation.top_improvements)
    if len(available) < require_at_least:
        raise ValueError(
            f"Need at least {require_at_least} driving examples to propose a "
            f"hypothesis; got {len(available)}. Try lowering "
            f"`require_at_least` or providing more eval data."
        )

    if require_inputs_and_outputs:
        for ex in available[:n_examples]:
            if ex.input is None or ex.old_output is None or ex.new_output is None:
                raise ValueError(
                    "Driving examples are missing input/old_output/new_output. "
                    "Pass these arrays to caliber.explain() so the judge has "
                    "something to read, or set require_inputs_and_outputs=False."
                )

    sample = available[: min(n_examples, _MAX_PROMPT_EXAMPLES)]
    prompt = _build_judge_prompt(explanation, sample)
    raw = provider.chat(prompt)

    return JudgedHypothesis(
        hypothesis=raw.strip(),
        provider=provider.name,
        model=provider.model,
        n_examples_reviewed=len(sample),
    )


def _build_judge_prompt(
    explanation: ExplanationResult, examples: list[ExampleFlip]
) -> str:
    """Construct the meta-prompt sent to the LLM judge."""
    parts: list[str] = [
        "You are a prompt-engineering analyst. Below are "
        f"{len(examples)} cases where prompt A produced a different answer "
        "than prompt B on the same input. The statistical analysis says "
        f"prompt B is {explanation.verdict} "
        f"(mean change {explanation.mean_difference:+.4f}, "
        f"95% CI [{explanation.ci_lower:+.4f}, {explanation.ci_upper:+.4f}], "
        f"n={explanation.n}).",
        "",
        "Read the cases. In 1-2 sentences, identify the pattern: what does "
        "prompt B do differently that explains the change? Be concrete; "
        "do not speculate beyond what the responses show.",
        "",
    ]
    for n, ex in enumerate(examples, start=1):
        parts.append(f"=== Case {n} ===")
        # input/old/new are guaranteed non-None when require_inputs_and_outputs=True
        parts.append(f"Input: {ex.input}")
        cat = f" [{ex.category}]" if ex.category else ""
        parts.append(f"Score change{cat}: {ex.old_score:.2f} -> {ex.new_score:.2f}")
        parts.append("")
        parts.append(f"Prompt A response:\n{_truncate(ex.old_output or '')}")
        parts.append("")
        parts.append(f"Prompt B response:\n{_truncate(ex.new_output or '')}")
        parts.append("")
    parts.append("Pattern (1-2 sentences):")
    return "\n".join(parts)


def _truncate(text: str, limit: int = _DEFAULT_OUTPUT_TRUNC) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 4] + " ..."
