"""
Review Confirmation Agent — Ollama edition.
BuiltInPlanner / ThinkingConfig removed (not supported by Ollama/LiteLLM).
"""

import logging

from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from pydantic import BaseModel, Field

from agentic_data_scientist.agents.adk.loop_detection import LoopDetectionAgent
from agentic_data_scientist.agents.adk.utils import REVIEW_MODEL, get_generate_content_config
from agentic_data_scientist.prompts import load_prompt


logger = logging.getLogger(__name__)


def _create_clear_decision_callback(state_key: str):
    def clear_decision_callback(callback_context: CallbackContext):
        ctx = callback_context._invocation_context
        state = ctx.session.state
        if state_key in state:
            del state[state_key]
    return clear_decision_callback


def _create_exit_loop_callback(state_key: str):
    def exit_loop_callback(callback_context: CallbackContext):
        ctx = callback_context._invocation_context
        state = ctx.session.state
        decision = state.get(state_key)

        if not decision or not isinstance(decision, dict):
            logger.warning(f"[ReviewConfirmation] No valid decision in '{state_key}' — not exiting loop")
            return None

        should_exit = decision.get("exit", False)
        reason = decision.get("reason", "No reason provided")

        if should_exit:
            logger.info(f"[ReviewConfirmation] Exiting loop (key='{state_key}') — {reason}")
            if hasattr(callback_context, '_event_actions') and callback_context._event_actions:
                callback_context._event_actions.escalate = True
            else:
                logger.warning("[ReviewConfirmation] No event_actions available — cannot escalate")
                return None
            return types.Content(role="model", parts=[])
        else:
            logger.info(f"[ReviewConfirmation] Continuing loop (key='{state_key}') — {reason}")
            return None

    return exit_loop_callback


class ReviewConfirmationOutput(BaseModel):
    exit: bool = Field(
        description="True to exit the review loop (approved), False to continue (needs more work)."
    )
    reason: str = Field(description="Brief explanation of the decision.")


REVIEW_CONFIRMATION_OUTPUT_SCHEMA = ReviewConfirmationOutput


def create_review_confirmation_agent(
    auto_exit_on_completion: bool = False,
    prompt_name: str = "plan_review_confirmation",
) -> LoopDetectionAgent:
    """Create a review confirmation agent (Ollama-compatible, no ThinkingConfig)."""
    logger.info(f"[AgenticDS] Creating review confirmation agent (prompt={prompt_name})")

    instruction = load_prompt(prompt_name)
    state_key = f"{prompt_name}_decision"

    before_callback = _create_clear_decision_callback(state_key)
    after_callback = _create_exit_loop_callback(state_key) if auto_exit_on_completion else None

    agent = LoopDetectionAgent(
        name=f"{prompt_name}_agent",
        model=REVIEW_MODEL,
        description="Determines whether to exit the review loop based on implementation status.",
        instruction=instruction,
        tools=[],
        generate_content_config=get_generate_content_config(temperature=0.0),
        output_schema=REVIEW_CONFIRMATION_OUTPUT_SCHEMA,
        output_key=state_key,
        before_agent_callback=before_callback,
        after_agent_callback=after_callback,
    )

    logger.info(
        f"[AgenticDS] Review confirmation agent created "
        f"(prompt={prompt_name}, state_key={state_key}, auto_exit={auto_exit_on_completion})"
    )
    return agent
