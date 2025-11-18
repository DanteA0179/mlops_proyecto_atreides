"""Parameter extraction from natural language queries."""

import re
from datetime import datetime, timedelta


class ParameterExtractor:
    """
    Extracts prediction parameters from natural language.

    Handles time expressions, load types, and numerical values
    to construct feature dictionaries for model predictions.
    """

    LOAD_TYPE_MAPPING = {
        "light": "Light",
        "medium": "Medium",
        "maximum": "Maximum",
        "max": "Maximum",
        "high": "Maximum",
        "low": "Light"
    }

    def extract(self, message: str) -> dict:
        """
        Extract prediction parameters from message.

        Parses natural language to identify time, day, and load type.
        Fills missing parameters with sensible defaults.

        Parameters
        ----------
        message : str
            User message containing prediction request

        Returns
        -------
        dict
            Extracted parameters with defaults for missing values

        Examples
        --------
        >>> extractor = ParameterExtractor()
        >>> params = extractor.extract("What will consumption be tomorrow at 10am with Medium load?")
        >>> 'nsm' in params
        True
        >>> params['load_type']
        'Medium'
        """
        params = {}

        # Extract time
        nsm = self._extract_time(message)
        if nsm is not None:
            params["nsm"] = nsm
        else:
            params["nsm"] = 36000  # Default: 10am

        # Extract day of week
        day_of_week = self._extract_day_of_week(message)
        params["day_of_week"] = day_of_week

        # Extract load type
        load_type = self._extract_load_type(message)
        params["load_type"] = load_type

        # Determine week status based on day
        params["week_status"] = "Weekday" if day_of_week < 5 else "Weekend"

        # Use default values for other parameters
        params.update({
            "lagging_reactive_power": 23.45,
            "leading_reactive_power": 12.30,
            "co2": 0.05,
            "lagging_power_factor": 0.85,
            "leading_power_factor": 0.92
        })

        return params

    def _extract_time(self, message: str) -> int | None:
        """
        Extract time and convert to NSM (seconds from midnight).

        Parameters
        ----------
        message : str
            User message

        Returns
        -------
        int | None
            Seconds from midnight, or None if no time found
        """
        message_lower = message.lower()

        # Pattern: "10am", "2pm", "14:00"
        time_pattern = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
        match = re.search(time_pattern, message_lower)

        if match:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0
            period = match.group(3)

            # Convert to 24-hour format
            if period == "pm" and hour != 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0

            # Convert to NSM
            nsm = hour * 3600 + minute * 60
            return nsm

        return None

    def _extract_day_of_week(self, message: str) -> int:
        """
        Extract day of week (0=Monday, 6=Sunday).

        Parameters
        ----------
        message : str
            User message

        Returns
        -------
        int
            Day of week number (0-6)
        """
        message_lower = message.lower()

        # Check for relative days
        if "tomorrow" in message_lower or "mañana" in message_lower:
            tomorrow = datetime.now() + timedelta(days=1)
            return tomorrow.weekday()
        elif "today" in message_lower or "hoy" in message_lower:
            return datetime.now().weekday()

        # Check for specific days
        days = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
            "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
            "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6
        }

        for day_name, day_num in days.items():
            if day_name in message_lower:
                return day_num

        # Default: Monday
        return 0

    def _extract_load_type(self, message: str) -> str:
        """
        Extract load type from message.

        Parameters
        ----------
        message : str
            User message

        Returns
        -------
        str
            Load type: Light, Medium, or Maximum
        """
        message_lower = message.lower()

        for keyword, load_type in self.LOAD_TYPE_MAPPING.items():
            if keyword in message_lower:
                return load_type

        # Default: Medium
        return "Medium"
