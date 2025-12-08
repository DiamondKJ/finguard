"""Dataset generation using Claude API for FinGuard."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from anthropic import Anthropic
from tqdm import tqdm

from src.utils.config import get_settings, load_config
from src.utils.logger import get_logger
from src.utils.validation import validate_prompt

logger = get_logger(__name__)


class PromptGenerator:
    """Generate labeled prompts using Claude API."""

    def __init__(
        self,
        config_path: str | Path = "config/dataset_config.yaml",
        api_key: str | None = None,
    ):
        """Initialize prompt generator.

        Args:
            config_path: Path to dataset configuration file
            api_key: Anthropic API key (if None, loads from settings)
        """
        self.config = load_config(config_path)
        self.settings = get_settings()

        api_key = api_key or self.settings.anthropic_api_key
        if not api_key:
            raise ValueError("Anthropic API key not provided")

        self.client = Anthropic(api_key=api_key)
        self.model = self.config["generation"]["model"]
        self.temperature = self.config["generation"]["temperature"]
        self.max_tokens = self.config["generation"]["max_tokens"]

        logger.info(
            "Initialized PromptGenerator",
            model=self.model,
            temperature=self.temperature,
        )

    def generate_for_category(
        self, category: Dict[str, Any], num_examples: int
    ) -> List[Dict[str, Any]]:
        """Generate examples for a specific category.

        Args:
            category: Category configuration dictionary
            num_examples: Number of examples to generate

        Returns:
            List of generated examples with labels
        """
        category_name = category["name"]
        label = category["label"]
        description = category["description"]
        seed_examples = category.get("examples", [])

        logger.info(
            f"Generating examples for category",
            category=category_name,
            num_examples=num_examples,
        )

        system_prompt = self._build_system_prompt(
            category_name, description, seed_examples
        )

        generated = []
        batch_size = self.config["generation"]["batch_size"]

        with tqdm(total=num_examples, desc=f"Generating {category_name}") as pbar:
            while len(generated) < num_examples:
                remaining = num_examples - len(generated)
                batch_count = min(batch_size, remaining)

                try:
                    batch = self._generate_batch(system_prompt, batch_count)
                    for prompt in batch:
                        try:
                            validated = validate_prompt(
                                prompt,
                                min_length=self.config["validation"]["min_length"],
                                max_length=self.config["validation"]["max_length"],
                            )
                            generated.append(
                                {
                                    "text": validated,
                                    "label": label,
                                    "category": category_name,
                                }
                            )
                            pbar.update(1)
                        except Exception as e:
                            logger.warning(
                                f"Skipping invalid prompt", error=str(e), prompt=prompt
                            )

                except Exception as e:
                    logger.error(f"Error generating batch", error=str(e))
                    time.sleep(2)  # Back off on error

                # Rate limiting
                time.sleep(0.5)

        logger.info(
            f"Generated examples for category",
            category=category_name,
            count=len(generated),
        )
        return generated

    def _build_system_prompt(
        self, category: str, description: str, examples: List[str]
    ) -> str:
        """Build system prompt for generation.

        Args:
            category: Category name
            description: Category description
            examples: Seed examples

        Returns:
            System prompt string
        """
        prompt = f"""You are a dataset generator for prompt safety classification.

Generate diverse, realistic prompts that fall into the category: {category}

Description: {description}

Here are some seed examples:
{chr(10).join(f"- {ex}" for ex in examples)}

Generate prompts that:
1. Are realistic variations of what real users might type
2. Cover different phrasings and approaches
3. Vary in length and complexity
4. Are unambiguous in their category
5. Don't duplicate the seed examples exactly

Generate prompts that clearly belong to this category, but with creative variation.
Each prompt should be on a new line. Generate exactly the requested number."""

        return prompt

    def _generate_batch(self, system_prompt: str, count: int) -> List[str]:
        """Generate a batch of prompts.

        Args:
            system_prompt: System prompt for generation
            count: Number of prompts to generate

        Returns:
            List of generated prompts
        """
        user_message = f"Generate {count} diverse prompts, one per line."

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        content = response.content[0].text
        prompts = [line.strip() for line in content.split("\n") if line.strip()]

        # Remove numbering if present (e.g., "1. ", "- ")
        prompts = [self._clean_prompt(p) for p in prompts]

        return prompts[:count]

    def _clean_prompt(self, prompt: str) -> str:
        """Clean generated prompt by removing numbering/bullets.

        Args:
            prompt: Raw generated prompt

        Returns:
            Cleaned prompt
        """
        # Remove common prefixes like "1. ", "- ", "* "
        import re

        prompt = re.sub(r"^[\d]+\.\s+", "", prompt)
        prompt = re.sub(r"^[-*]\s+", "", prompt)
        return prompt.strip()

    def generate_full_dataset(self) -> List[Dict[str, Any]]:
        """Generate complete dataset for all categories.

        Returns:
            List of all generated examples
        """
        examples_per_category = self.config["dataset"]["examples_per_category"]
        all_examples = []

        for category in self.config["categories"]:
            category_examples = self.generate_for_category(
                category, examples_per_category
            )
            all_examples.extend(category_examples)

        logger.info(f"Generated full dataset", total_examples=len(all_examples))
        return all_examples

    def save_dataset(
        self, examples: List[Dict[str, Any]], output_path: str | Path
    ) -> None:
        """Save generated dataset to JSON file.

        Args:
            examples: List of generated examples
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(examples, f, indent=2)

        logger.info(f"Saved dataset", path=str(output_path), count=len(examples))


if __name__ == "__main__":
    # Test generator
    from src.utils.logger import setup_logging

    setup_logging(log_level="INFO")

    try:
        generator = PromptGenerator()
        logger.info("Generator initialized successfully")

        # Generate small test batch for one category
        test_config = load_config("config/dataset_config.yaml")
        category = test_config["categories"][0]  # SAFE category

        examples = generator.generate_for_category(category, num_examples=5)
        print(f"\nGenerated {len(examples)} examples:")
        for ex in examples:
            print(f"  [{ex['category']}] {ex['text']}")

    except Exception as e:
        logger.error("Error testing generator", error=str(e), exc_info=True)
