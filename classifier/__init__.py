"""
LLM Text Classifier Package

A high-performance, scalable microservice for text classification using Large Language Models.
Supports multiple classification tasks with configurable prompts and various LLM providers.
"""

from .text_classifier import TextClassifier
from .config_manager import ConfigManager
from .gemini_service import GeminiService

__version__ = "1.0.0"
__author__ = "LLM Classifier Team"

__all__ = [
    "TextClassifier",
    "ConfigManager", 
    "GeminiService"
]
