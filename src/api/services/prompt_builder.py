"""Prompt construction for LLM interactions."""

from pathlib import Path
from typing import Literal

Intent = Literal["prediction", "analysis", "what-if", "explanation", "general"]


class PromptBuilder:
    """
    Builds prompts for LLM based on intent and context.

    Loads system prompts from config files and constructs
    complete message sequences for LLM API calls.
    """

    def __init__(self, prompts_dir: Path | None = None):
        """
        Initialize PromptBuilder.

        Parameters
        ----------
        prompts_dir : Path | None
            Directory containing prompt template files
        """
        if prompts_dir is None:
            prompts_dir = Path("config/prompts")
        self.prompts_dir = prompts_dir
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load system prompt from file."""
        system_prompt_path = self.prompts_dir / "system_prompt.txt"
        if system_prompt_path.exists():
            return system_prompt_path.read_text(encoding="utf-8")
        return self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """Default system prompt if file not found."""
        return """You are an AI assistant specialized in energy optimization for the steel industry.

Your role:
- Help operators understand energy consumption patterns
- Provide predictions based on ML models
- Explain factors affecting consumption
- Suggest optimization strategies

Guidelines:
- Be concise and technical but accessible
- Use specific numbers when available
- Explain your reasoning
- If you don't have data, say so clearly
- Focus on actionable insights

Context:
- Steel plant in South Korea
- Data from 2018, 15-minute intervals
- Key factors: reactive power, CO2, power factor, load type, time
- Target: Usage_kWh (energy consumption)
"""

    def build_prompt(
        self,
        intent: Intent,
        message: str,
        context: list[dict] | None = None
    ) -> list[dict]:
        """
        Build prompt messages for LLM.

        Constructs a complete message sequence including system prompt,
        intent-specific instructions, conversation context, and user message.

        Parameters
        ----------
        intent : Intent
            Detected intent category
        message : str
            Current user message
        context : list[dict] | None
            Previous conversation messages

        Returns
        -------
        list[dict]
            Messages in format [{"role": "system", "content": "..."}, ...]

        Examples
        --------
        >>> builder = PromptBuilder()
        >>> messages = builder.build_prompt("prediction", "What will consumption be tomorrow?")
        >>> len(messages) >= 2
        True
        """
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Add intent-specific instructions
        intent_instruction = self._get_intent_instruction(intent)
        if intent_instruction:
            messages.append({
                "role": "system",
                "content": intent_instruction
            })

        # Add conversation context
        if context:
            messages.extend(context[-10:])  # Last 10 messages

        # Add current user message
        messages.append({
            "role": "user",
            "content": message
        })

        return messages

    def _get_intent_instruction(self, intent: Intent) -> str | None:
        """
        Get intent-specific instruction.

        Parameters
        ----------
        intent : Intent
            Detected intent category

        Returns
        -------
        str | None
            Intent-specific instruction or None for general intent
        """
        instructions = {
            "prediction": "The user wants a prediction. Extract parameters and provide a forecast based on the model results.",
            "what-if": "The user wants to explore a scenario. Compare current vs proposed situation and estimate impact.",
            "analysis": "The user wants to understand why something happened. Analyze patterns and provide insights.",
            "explanation": "The user wants to learn about concepts. Explain clearly with examples.",
            "general": None
        }
        return instructions.get(intent)
