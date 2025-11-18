"""Unit tests for ParameterExtractor."""

import pytest
from src.api.services.parameter_extractor import ParameterExtractor


class TestParameterExtractor:
    """Tests for ParameterExtractor service."""

    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = ParameterExtractor()

    def test_extract_time_am(self):
        """Test extraction of AM times."""
        test_cases = [
            ("consumption at 10am", 36000),  # 10:00 AM = 10*3600
            ("usage at 9am", 32400),  # 9:00 AM = 9*3600
            ("at 12am", 0),  # 12:00 AM = midnight
        ]

        for message, expected_nsm in test_cases:
            params = self.extractor.extract(message)
            assert params["nsm"] == expected_nsm, f"Failed for: {message}"

    def test_extract_time_pm(self):
        """Test extraction of PM times."""
        test_cases = [
            ("consumption at 2pm", 50400),  # 2:00 PM = 14*3600
            ("usage at 6pm", 64800),  # 6:00 PM = 18*3600
            ("at 12pm", 43200),  # 12:00 PM = noon
        ]

        for message, expected_nsm in test_cases:
            params = self.extractor.extract(message)
            assert params["nsm"] == expected_nsm, f"Failed for: {message}"

    def test_extract_time_with_minutes(self):
        """Test extraction of times with minutes."""
        message = "consumption at 10:30am"
        params = self.extractor.extract(message)
        # 10:30 AM = 10*3600 + 30*60 = 37800
        assert params["nsm"] == 37800

    def test_extract_day_tomorrow(self):
        """Test extraction of 'tomorrow'."""
        from datetime import datetime, timedelta

        message = "consumption tomorrow"
        params = self.extractor.extract(message)

        expected_day = (datetime.now() + timedelta(days=1)).weekday()
        assert params["day_of_week"] == expected_day

    def test_extract_day_today(self):
        """Test extraction of 'today'."""
        from datetime import datetime

        message = "consumption today"
        params = self.extractor.extract(message)

        expected_day = datetime.now().weekday()
        assert params["day_of_week"] == expected_day

    def test_extract_specific_day(self):
        """Test extraction of specific days."""
        test_cases = [
            ("consumption on Monday", 0),
            ("usage on Tuesday", 1),
            ("on Wednesday", 2),
            ("Thursday", 3),
            ("Friday", 4),
            ("Saturday", 5),
            ("Sunday", 6),
        ]

        for message, expected_day in test_cases:
            params = self.extractor.extract(message)
            assert params["day_of_week"] == expected_day, f"Failed for: {message}"

    def test_extract_load_type_light(self):
        """Test extraction of Light load type."""
        test_cases = [
            "consumption with light load",
            "usage with low load",
        ]

        for message in test_cases:
            params = self.extractor.extract(message)
            assert params["load_type"] == "Light", f"Failed for: {message}"

    def test_extract_load_type_medium(self):
        """Test extraction of Medium load type."""
        message = "consumption with medium load"
        params = self.extractor.extract(message)
        assert params["load_type"] == "Medium"

    def test_extract_load_type_maximum(self):
        """Test extraction of Maximum load type."""
        test_cases = [
            "consumption with maximum load",
            "usage with max load",
            "with high load",
        ]

        for message in test_cases:
            params = self.extractor.extract(message)
            assert params["load_type"] == "Maximum", f"Failed for: {message}"

    def test_extract_defaults(self):
        """Test default values when nothing is specified."""
        message = "what will consumption be"
        params = self.extractor.extract(message)

        # Check defaults
        assert params["nsm"] == 36000  # 10am
        assert params["day_of_week"] == 0  # Monday
        assert params["load_type"] == "Medium"
        assert params["week_status"] == "Weekday"

    def test_extract_week_status_weekday(self):
        """Test week status for weekdays."""
        test_cases = [
            "consumption on Monday",
            "on Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
        ]

        for message in test_cases:
            params = self.extractor.extract(message)
            assert params["week_status"] == "Weekday", f"Failed for: {message}"

    def test_extract_week_status_weekend(self):
        """Test week status for weekend."""
        test_cases = [
            "consumption on Saturday",
            "on Sunday",
        ]

        for message in test_cases:
            params = self.extractor.extract(message)
            assert params["week_status"] == "Weekend", f"Failed for: {message}"

    def test_extract_complete_query(self):
        """Test extraction from complete query."""
        message = "What will be the consumption tomorrow at 10am with Medium load?"
        params = self.extractor.extract(message)

        assert params["nsm"] == 36000  # 10am
        assert params["load_type"] == "Medium"
        assert "lagging_reactive_power" in params
        assert "leading_reactive_power" in params
        assert "co2" in params

    def test_extract_spanish_keywords(self):
        """Test extraction with Spanish keywords."""
        test_cases = [
            ("consumo mañana", "day_of_week"),  # tomorrow
            ("consumo el lunes", 0),  # Monday
            ("el martes", 1),  # Tuesday
        ]

        from datetime import datetime, timedelta

        # Test 'mañana' (tomorrow)
        message = "consumo mañana"
        params = self.extractor.extract(message)
        expected_day = (datetime.now() + timedelta(days=1)).weekday()
        assert params["day_of_week"] == expected_day

        # Test specific days
        day_tests = [
            ("el lunes", 0),
            ("el martes", 1),
            ("el viernes", 4),
        ]

        for msg, expected_day in day_tests:
            params = self.extractor.extract(msg)
            assert params["day_of_week"] == expected_day, f"Failed for: {msg}"
