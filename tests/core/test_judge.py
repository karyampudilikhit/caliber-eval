"""Tests for caliber.core.judge — the LLM-judged hypothesis layer.

Uses a deterministic MockProvider so no real LLM call happens in tests.
The OllamaProvider is tested only at the import / construction level — we
don't hit a real Ollama server in the test suite.
"""

from __future__ import annotations

from typing import cast

import pytest

from caliber import (
    ExplanationResult,
    JudgedHypothesis,
    OllamaProvider,
    explain,
    judge_hypothesis,
)
from caliber.core.judge import LLMProvider


class MockProvider:
    """Deterministic in-memory provider used for testing.

    Records the prompts it received so tests can assert on them.
    """

    name = "mock"

    def __init__(self, response: str = "Mock hypothesis: B uses CoT, A does not.") -> None:
        self.model = "mock-v1"
        self.response = response
        self.prompts: list[str] = []

    def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


@pytest.fixture
def explanation_with_outputs() -> ExplanationResult:
    """An explanation with inputs + old_outputs + new_outputs populated."""
    return explain(
        old_scores=[0, 0, 0, 1, 1, 0, 0],
        new_scores=[1, 1, 1, 1, 1, 1, 1],
        inputs=[
            "Compute 8 + 4*3 - 2",
            "If 3x+7=25, what is x?",
            "What is 13 squared?",
            "What is 5^4?",
            "What is sqrt(256)?",
            "Compute (12+6)*4-10",
            "If 2(x+4)=26, what is x?",
        ],
        old_outputs=[
            "16",
            "x = 4",
            "169",
            "625",
            "16",
            "58",
            "x = 5",
        ],
        new_outputs=[
            "Following PEMDAS: 4*3=12, then 8+12-2=18. ANSWER = 18",
            "Subtract 7: 3x=18. Divide by 3: x=6. ANSWER = 6",
            "13*13=169. ANSWER = 169",
            "5*5*5*5=625. ANSWER = 625",
            "16*16=256, so sqrt(256)=16. ANSWER = 16",
            "(12+6)=18, 18*4=72, 72-10=62. ANSWER = 62",
            "Distribute: 2x+8=26. Subtract 8: 2x=18. Divide: x=9. ANSWER = 9",
        ],
        seed=0,
    )


# ============================================================================
# Return shape
# ============================================================================


class TestReturnShape:
    def test_returns_judged_hypothesis(
        self, explanation_with_outputs: ExplanationResult
    ) -> None:
        provider = MockProvider()
        result = judge_hypothesis(
            explanation_with_outputs, cast(LLMProvider, provider)
        )
        assert isinstance(result, JudgedHypothesis)

    def test_hypothesis_text_from_provider(
        self, explanation_with_outputs: ExplanationResult
    ) -> None:
        provider = MockProvider(response="Custom hypothesis text.")
        result = judge_hypothesis(
            explanation_with_outputs, cast(LLMProvider, provider)
        )
        assert result.hypothesis == "Custom hypothesis text."

    def test_provider_metadata_recorded(
        self, explanation_with_outputs: ExplanationResult
    ) -> None:
        provider = MockProvider()
        result = judge_hypothesis(
            explanation_with_outputs, cast(LLMProvider, provider)
        )
        assert result.provider == "mock"
        assert result.model == "mock-v1"
        assert result.is_ai_generated is True

    def test_n_examples_reviewed_matches_request(
        self, explanation_with_outputs: ExplanationResult
    ) -> None:
        # explanation has at least 3 driving improvements
        provider = MockProvider()
        result = judge_hypothesis(
            explanation_with_outputs, cast(LLMProvider, provider), n_examples=3
        )
        assert result.n_examples_reviewed == 3

    def test_hypothesis_is_stripped(
        self, explanation_with_outputs: ExplanationResult
    ) -> None:
        provider = MockProvider(response="   leading and trailing whitespace   \n")
        result = judge_hypothesis(
            explanation_with_outputs, cast(LLMProvider, provider)
        )
        assert result.hypothesis == "leading and trailing whitespace"


# ============================================================================
# Prompt construction
# ============================================================================


class TestPromptConstruction:
    def test_provider_called_exactly_once(
        self, explanation_with_outputs: ExplanationResult
    ) -> None:
        provider = MockProvider()
        judge_hypothesis(explanation_with_outputs, cast(LLMProvider, provider))
        assert len(provider.prompts) == 1

    def test_prompt_mentions_verdict(
        self, explanation_with_outputs: ExplanationResult
    ) -> None:
        provider = MockProvider()
        judge_hypothesis(explanation_with_outputs, cast(LLMProvider, provider))
        prompt = provider.prompts[0]
        assert explanation_with_outputs.verdict in prompt

    def test_prompt_includes_inputs(
        self, explanation_with_outputs: ExplanationResult
    ) -> None:
        provider = MockProvider()
        judge_hypothesis(explanation_with_outputs, cast(LLMProvider, provider))
        prompt = provider.prompts[0]
        # At least the input from the first improvement should appear
        first_input = explanation_with_outputs.top_improvements[0].input
        assert first_input is not None
        assert first_input in prompt

    def test_prompt_includes_both_responses(
        self, explanation_with_outputs: ExplanationResult
    ) -> None:
        provider = MockProvider()
        judge_hypothesis(explanation_with_outputs, cast(LLMProvider, provider))
        prompt = provider.prompts[0]
        first = explanation_with_outputs.top_improvements[0]
        assert first.old_output is not None
        assert first.new_output is not None
        assert first.old_output in prompt
        assert first.new_output in prompt

    def test_prompt_includes_n_cases(
        self, explanation_with_outputs: ExplanationResult
    ) -> None:
        provider = MockProvider()
        judge_hypothesis(
            explanation_with_outputs, cast(LLMProvider, provider), n_examples=2
        )
        prompt = provider.prompts[0]
        assert "Case 1" in prompt
        assert "Case 2" in prompt
        assert "Case 3" not in prompt

    def test_long_outputs_truncated(self) -> None:
        # Build an explanation with very long outputs to verify truncation
        long_output = "x" * 2000
        exp = explain(
            old_scores=[0, 0],
            new_scores=[1, 1],
            inputs=["q1", "q2"],
            old_outputs=["short_old", "short_old"],
            new_outputs=[long_output, long_output],
            seed=0,
        )
        provider = MockProvider()
        judge_hypothesis(exp, cast(LLMProvider, provider))
        prompt = provider.prompts[0]
        # Truncation marker "..." should appear when output is long.
        assert "..." in prompt
        # Full 2000 chars should NOT appear.
        assert long_output not in prompt


# ============================================================================
# Validation
# ============================================================================


class TestValidation:
    def test_too_few_examples_raises(self) -> None:
        # Only one improvement available
        exp = explain(
            old_scores=[0, 1, 1, 1],
            new_scores=[1, 1, 1, 1],
            inputs=["q1", "q2", "q3", "q4"],
            old_outputs=["a", "b", "c", "d"],
            new_outputs=["w", "x", "y", "z"],
            seed=0,
        )
        provider = MockProvider()
        with pytest.raises(ValueError, match="at least"):
            judge_hypothesis(
                exp, cast(LLMProvider, provider), require_at_least=2
            )

    def test_missing_inputs_raises(self) -> None:
        # No inputs provided to explain()
        exp = explain(
            old_scores=[0, 0, 0],
            new_scores=[1, 1, 1],
            seed=0,
        )
        provider = MockProvider()
        with pytest.raises(ValueError, match="input"):
            judge_hypothesis(exp, cast(LLMProvider, provider))

    def test_missing_outputs_raises(self) -> None:
        exp = explain(
            old_scores=[0, 0, 0],
            new_scores=[1, 1, 1],
            inputs=["q1", "q2", "q3"],
            # no old_outputs / new_outputs
            seed=0,
        )
        provider = MockProvider()
        with pytest.raises(ValueError, match=r"input|output"):
            judge_hypothesis(exp, cast(LLMProvider, provider))

    def test_can_disable_io_requirement(self) -> None:
        # With the safety flag off, even score-only explanations are allowed.
        exp = explain(
            old_scores=[0, 0, 0],
            new_scores=[1, 1, 1],
            seed=0,
        )
        provider = MockProvider()
        result = judge_hypothesis(
            exp,
            cast(LLMProvider, provider),
            require_inputs_and_outputs=False,
        )
        # The provider was still called — no exception.
        assert result.hypothesis == "Mock hypothesis: B uses CoT, A does not."


# ============================================================================
# Provider protocol
# ============================================================================


class TestProviderProtocol:
    def test_mock_satisfies_llm_provider_protocol(self) -> None:
        provider = MockProvider()
        # runtime_checkable means isinstance works on the Protocol
        assert isinstance(provider, LLMProvider)

    def test_ollama_provider_has_correct_attributes(self) -> None:
        # We don't actually call .chat() — just verify the construct API.
        provider = OllamaProvider(model="phi3")
        assert provider.name == "ollama"
        assert provider.model == "phi3"
        assert callable(provider.chat)

    def test_ollama_provider_default_model(self) -> None:
        provider = OllamaProvider()
        assert provider.model == "phi3"

    def test_ollama_provider_custom_url(self) -> None:
        provider = OllamaProvider(url="http://example.com:9999/api/generate")
        assert provider.url == "http://example.com:9999/api/generate"
