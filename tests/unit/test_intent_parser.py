"""Unit tests for IntentParser."""

import pytest
from src.api.services.intent_parser import IntentParser


class TestIntentParser:
    """Tests for IntentParser service."""

    def setup_method(self):
        """Setup test fixtures."""
        self.parser = IntentParser()

    def test_parse_prediction_intent_english(self):
        """Test detection of prediction intent in English."""
        messages = [
            "What will be the consumption tomorrow?",
            "Predict usage at 10am",
            "How much energy will we use next Monday?",
            "Estimate consumption for tomorrow at 2pm",
            "What will be the usage tomorrow"
        ]

        for msg in messages:
            intent = self.parser.parse(msg)
            assert intent == "prediction", f"Failed for: {msg}"

    def test_parse_prediction_intent_spanish(self):
        """Test detection of prediction intent in Spanish."""
        messages = [
            "Cuánto será el consumo mañana?",
            "Qué será el uso de energía el lunes?"
        ]

        for msg in messages:
            intent = self.parser.parse(msg)
            assert intent == "prediction", f"Failed for: {msg}"

    def test_parse_whatif_intent_english(self):
        """Test detection of what-if intent in English."""
        messages = [
            "What if I shift production to 2am?",
            "How much would I save if I change the load?",
            "Suppose we reduce the power factor",
            "If I modify the schedule what happens?",
            "How can I optimize consumption?"
        ]

        for msg in messages:
            intent = self.parser.parse(msg)
            assert intent == "what-if", f"Failed for: {msg}"

    def test_parse_whatif_intent_spanish(self):
        """Test detection of what-if intent in Spanish."""
        messages = [
            "Qué pasa si cambio la producción a las 2am?",
            "Cuánto ahorraré si reduzco la carga?"
        ]

        for msg in messages:
            intent = self.parser.parse(msg)
            assert intent == "what-if", f"Failed for: {msg}"

    def test_parse_analysis_intent_english(self):
        """Test detection of analysis intent in English."""
        messages = [
            "Why was there a spike last Friday?",
            "Analyze the consumption pattern",
            "What caused the high usage?",
            "Explain the reason for the peak",
            "Why is consumption low today?"
        ]

        for msg in messages:
            intent = self.parser.parse(msg)
            assert intent == "analysis", f"Failed for: {msg}"

    def test_parse_analysis_intent_spanish(self):
        """Test detection of analysis intent in Spanish."""
        messages = [
            "Por qué hubo un pico el viernes pasado?",
            "Cuál es la causa del alto consumo?"
        ]

        for msg in messages:
            intent = self.parser.parse(msg)
            assert intent == "analysis", f"Failed for: {msg}"

    def test_parse_explanation_intent_english(self):
        """Test detection of explanation intent in English."""
        messages = [
            "What is power factor?",
            "Explain reactive power",
            "How does load type affect consumption?",
            "What factors influence energy usage?",
            "Define lagging power factor"
        ]

        for msg in messages:
            intent = self.parser.parse(msg)
            assert intent == "explanation", f"Failed for: {msg}"

    def test_parse_explanation_intent_spanish(self):
        """Test detection of explanation intent in Spanish."""
        messages = [
            "Qué es el factor de potencia?",
            "Cómo funciona la potencia reactiva?"
        ]

        for msg in messages:
            intent = self.parser.parse(msg)
            assert intent == "explanation", f"Failed for: {msg}"

    def test_parse_general_intent(self):
        """Test fallback to general intent."""
        messages = [
            "Hello",
            "Thanks",
            "Random text without keywords",
            "Goodbye",
            "OK"
        ]

        for msg in messages:
            intent = self.parser.parse(msg)
            assert intent == "general", f"Failed for: {msg}"

    def test_parse_empty_message(self):
        """Test handling of empty messages."""
        intent = self.parser.parse("")
        assert intent == "general"

    def test_parse_case_insensitive(self):
        """Test case-insensitive matching."""
        messages = [
            "WHAT WILL BE THE CONSUMPTION?",
            "what will be the consumption?",
            "What Will Be The Consumption?"
        ]

        for msg in messages:
            intent = self.parser.parse(msg)
            assert intent == "prediction"
