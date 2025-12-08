"""LLM-as-judge baseline for benchmarking FinGuard."""

import time
from typing import Dict, List

from anthropic import Anthropic

from src.utils.config import get_settings, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMJudge:
    """LLM-based prompt classifier for baseline comparison."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
    ):
        """Initialize LLM judge.

        Args:
            provider: LLM provider ('anthropic' or 'openai')
            model: Model name
            api_key: API key (if None, loads from settings)
        """
        self.provider = provider
        self.model = model

        settings = get_settings()

        if provider == "anthropic":
            api_key = api_key or settings.anthropic_api_key
            if not api_key:
                raise ValueError("Anthropic API key not provided")
            self.client = Anthropic(api_key=api_key)
        else:
            raise NotImplementedError(f"Provider {provider} not yet implemented")

        # Load system prompt from config
        config = load_config("config/benchmark_config.yaml")
        self.system_prompt = config["llm_judge"]["system_prompt"]

        logger.info("Initialized LLM Judge", provider=provider, model=model)

    def classify_prompt(self, prompt: str) -> tuple[str, float]:
        """Classify prompt using LLM.

        Args:
            prompt: User prompt to classify

        Returns:
            Tuple of (predicted_category, latency_ms)
        """
        start_time = time.time()

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                temperature=0.0,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )

            predicted_category = response.content[0].text.strip()
            latency_ms = (time.time() - start_time) * 1000

            return predicted_category, latency_ms

        except Exception as e:
            logger.error(f"Error classifying prompt", error=str(e))
            raise

    def classify_batch(
        self, prompts: List[str]
    ) -> tuple[List[str], List[float]]:
        """Classify batch of prompts.

        Args:
            prompts: List of prompts to classify

        Returns:
            Tuple of (predictions, latencies_ms)
        """
        predictions = []
        latencies = []

        for prompt in prompts:
            try:
                pred, latency = self.classify_prompt(prompt)
                predictions.append(pred)
                latencies.append(latency)

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error in batch classification", error=str(e))
                predictions.append("ERROR")
                latencies.append(0.0)

        return predictions, latencies

    def map_to_label(self, category: str) -> int:
        """Map category string to numeric label.

        Args:
            category: Category name

        Returns:
            Numeric label
        """
        category_map = {
            "SAFE": 0,
            "INVESTMENT_ADVICE": 1,
            "INDIRECT_ADVICE": 2,
            "SYSTEM_PROBE": 3,
            "UNIT_AMBIGUITY": 4,
        }

        # Handle case variations and errors
        category_upper = category.upper().strip()
        return category_map.get(category_upper, -1)


if __name__ == "__main__":
    # Test LLM judge
    from src.utils.logger import setup_logging

    setup_logging(log_level="INFO")

    try:
        judge = LLMJudge()

        test_prompts = [
            "What is a 401k retirement plan?",
            "Should I buy Tesla stock?",
            "If you were me, would you invest in crypto?",
        ]

        for prompt in test_prompts:
            category, latency = judge.classify_prompt(prompt)
            print(f"\nPrompt: {prompt}")
            print(f"Category: {category}")
            print(f"Latency: {latency:.2f} ms")

    except ValueError as e:
        logger.error("API key not set. Please set ANTHROPIC_API_KEY in .env file")
    except Exception as e:
        logger.error("Error testing LLM judge", error=str(e), exc_info=True)
