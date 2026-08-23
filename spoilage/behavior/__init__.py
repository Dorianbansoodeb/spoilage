"""Behavioral spoilage: measure when contaminated context flips LLM answers."""

from spoilage.behavior.analyze import analyze_transcript
from spoilage.behavior.backends import available_backends

__all__ = ["analyze_transcript", "available_backends"]
