"""
Utility functions and configurations for ADK agents.
Configured for LOCAL OLLAMA models — no API keys required.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.genai import types


load_dotenv()

logger = logging.getLogger(__name__)

# Ollama base URL
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model names (LiteLLM uses "ollama/<model>" as the model string)
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL", "ollama/qwen3:8b")
REVIEW_MODEL_NAME  = os.getenv("REVIEW_MODEL",  "ollama/gemma4:latest")
CODING_MODEL_NAME  = os.getenv("CODING_MODEL",  "ollama/gpt-oss:120b-cloud")

logger.info(f"[AgenticDS] DEFAULT_MODEL={DEFAULT_MODEL_NAME}")
logger.info(f"[AgenticDS] REVIEW_MODEL={REVIEW_MODEL_NAME}")
logger.info(f"[AgenticDS] CODING_MODEL={CODING_MODEL_NAME}")
logger.info(f"[AgenticDS] OLLAMA_BASE_URL={OLLAMA_BASE_URL}")

# LiteLLM model instances pointing at local Ollama
DEFAULT_MODEL = LiteLlm(
    model=DEFAULT_MODEL_NAME,
    num_retries=3,
    timeout=300,
    api_base=OLLAMA_BASE_URL,
)

REVIEW_MODEL = LiteLlm(
    model=REVIEW_MODEL_NAME,
    num_retries=3,
    timeout=300,
    api_base=OLLAMA_BASE_URL,
)

LANGUAGE_REQUIREMENT = ""

__all__ = [
    "DEFAULT_MODEL",
    "REVIEW_MODEL",
    "DEFAULT_MODEL_NAME",
    "REVIEW_MODEL_NAME",
    "CODING_MODEL_NAME",
    "OLLAMA_BASE_URL",
    "get_generate_content_config",
    "exit_loop_simple",
    "is_network_disabled",
]


def is_network_disabled() -> bool:
    return os.getenv("DISABLE_NETWORK_ACCESS", "").lower() in ("true", "1")


def exit_loop_simple(tool_context: ToolContext):
    """DEPRECATED — use review_confirmation agents."""
    tool_context.actions.escalate = True
    return {}


def get_generate_content_config(temperature: float = 0.0, output_tokens: Optional[int] = None):
    """
    Create a GenerateContentConfig for Ollama models.
    ThinkingConfig is intentionally omitted — not supported by Ollama.
    """
    return types.GenerateContentConfig(
        temperature=temperature,
        top_p=0.95,
        seed=42,
        max_output_tokens=output_tokens,
    )
