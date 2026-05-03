"""
Implementation loop agents — Ollama edition.
ClaudeCodeAgent replaced with OllamaCodingAgent.
BuiltInPlanner / ThinkingConfig removed.
"""

import logging

from google.adk.tools.tool_context import CallbackContext
from google.genai import types

from agentic_data_scientist.agents.adk.event_compression import create_compression_callback
from agentic_data_scientist.agents.adk.loop_detection import LoopDetectionAgent
from agentic_data_scientist.agents.adk.review_confirmation import create_review_confirmation_agent
from agentic_data_scientist.agents.adk.utils import REVIEW_MODEL, get_generate_content_config
from agentic_data_scientist.prompts import load_prompt


logger = logging.getLogger(__name__)

TOOL_LOOP_LIMIT = 5
CODING_EVENT_LIMIT = 100
MAX_EVENTS_TO_KEEP = 20


def trim_history_to_recent_events(callback_context: CallbackContext, max_events: int = CODING_EVENT_LIMIT):
    session = callback_context._invocation_context.session
    events = session.events
    if len(events) > max_events:
        events_to_remove = len(events) - max_events
        for _ in range(events_to_remove):
            events.pop(0)


def make_implementation_agents(working_dir: str, tools: list):
    """Create coding + review + confirmation agents using local Ollama."""
    logger.info(f"[AgenticDS] Initializing implementation agents with {len(tools)} tools")

    # Use OllamaCodingAgent instead of ClaudeCodeAgent
    from agentic_data_scientist.agents.ollama_coding import OllamaCodingAgent

    coding_compression_callback = create_compression_callback(event_threshold=40, overlap_size=20)

    coding_agent = OllamaCodingAgent(
        name="coding_agent",
        description="A coding agent that implements plans using local Ollama.",
        working_dir=working_dir,
        output_key="implementation_summary",
        tools=tools,
        after_agent_callback=coding_compression_callback,
    )

    # Review agent (no ThinkingConfig for Ollama)
    review_prompt = load_prompt("coding_review")
    review_compression_callback = create_compression_callback(event_threshold=40, overlap_size=20)

    review_agent = LoopDetectionAgent(
        name="review_agent",
        description="Reviews implementation and provides feedback or approval.",
        instruction=review_prompt,
        model=REVIEW_MODEL,
        tools=tools,
        generate_content_config=get_generate_content_config(temperature=0.0),
        output_key="review_feedback",
        include_contents="none",
        after_agent_callback=review_compression_callback,
    )

    logger.info("[AgenticDS] Implementation agents created successfully")

    return (
        coding_agent,
        review_agent,
        create_review_confirmation_agent(
            auto_exit_on_completion=True, prompt_name="implementation_review_confirmation"
        ),
    )
