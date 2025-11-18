"""Intent detection from natural language queries."""

import re
from typing import Literal

Intent = Literal["prediction", "analysis", "what-if", "explanation", "general"]


class IntentParser:
    """
    Parses user message to detect intent.

    Uses keyword matching and regex patterns to classify user queries
    into one of five intents: prediction, analysis, what-if, explanation, or general.
    """

    INTENT_PATTERNS = {
        "prediction": [
            r"\b(predict|forecast|estimate|will be)\b",
            r"\b(tomorrow|next|future|at \d+am|at \d+pm)\b",
            r"\b(how much will|what will|cuánto será|qué será)\b"
        ],
        "what-if": [
            r"\b(what if|if I|if we|suppose|scenario|cambio)\b",
            r"\b(save|shift|modify|adjust|cambiar|ahorrar|reducir|reduzco)\b",
            r"\b(would.*save|would.*reduce|can.*optimize|ahorraré)\b"
        ],
        "analysis": [
            r"\b(why|reason|cause|caused|por qué|causa)\b",
            r"\b(spike|peak|pattern)\b",
            r"\b(analyze\b|analysis\b|trend|patrón)\b"
        ],
        "explanation": [
            r"\b(explain|what is|how does|qué es|cómo funciona)\b",
            r"\b(factors? affect|factors? influence|afecta)\b",
            r"\b(definition|meaning|significado|define)\b"
        ]
    }

    def parse(self, message: str) -> Intent:
        """
        Detect intent from user message.

        Uses pattern matching to score each intent category, returning
        the highest scoring intent or 'general' if no patterns match.

        Parameters
        ----------
        message : str
            User query in natural language

        Returns
        -------
        Intent
            Detected intent: prediction, analysis, what-if, explanation, or general

        Examples
        --------
        >>> parser = IntentParser()
        >>> parser.parse("What will be the consumption tomorrow?")
        'prediction'
        >>> parser.parse("Why was there a spike last Friday?")
        'analysis'
        """
        message_lower = message.lower()

        # Score each intent
        scores = dict.fromkeys(self.INTENT_PATTERNS.keys(), 0)

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    scores[intent] += 1

        # Get intent with highest score
        max_score = max(scores.values())

        if max_score == 0:
            return "general"

        # Return intent with highest score
        for intent, score in scores.items():
            if score == max_score:
                return intent

        return "general"
