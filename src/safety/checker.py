from __future__ import annotations

import re

from src.models import SafetyResult


DOSAGE_PATTERN = re.compile(r"\b(take|use|dose)\b.{0,40}\b\d+\s?(mg|ml|g)\b", re.IGNORECASE)
PRESCRIPTION_PATTERN = re.compile(r"\b(ibuprofen|amoxicillin|metformin|prednisone)\b", re.IGNORECASE)
SELF_HARM_PATTERN = re.compile(r"\b(self-harm|suicide|harm yourself)\b", re.IGNORECASE)


class SafetyChecker:
    def check(self, response: str) -> SafetyResult:
        flagged = []
        if DOSAGE_PATTERN.search(response):
            flagged.append("dosage")
        if PRESCRIPTION_PATTERN.search(response) and DOSAGE_PATTERN.search(response):
            flagged.append("prescription_dosage")
        if SELF_HARM_PATTERN.search(response):
            flagged.append("self_harm")
        if flagged:
            return SafetyResult(safe=False, reason="Unsafe medical advice detected", flagged_patterns=flagged)
        return SafetyResult(safe=True)

