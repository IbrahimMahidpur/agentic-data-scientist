"""
Event compression system using LLM-based summarization.
Ollama edition — all OpenRouter/cloud references removed.
"""

import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event, EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.genai import types as genai_types

from agentic_data_scientist.agents.adk.utils import DEFAULT_MODEL_NAME, OLLAMA_BASE_URL


logger = logging.getLogger(__name__)

DEFAULT_EVENT_THRESHOLD = 40
DEFAULT_OVERLAP_SIZE = 20
LARGE_TEXT_THRESHOLD = 10000
LARGE_TEXT_KEEP = 1000


def _truncate_large_event_texts(events: list) -> None:
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    text_len = len(part.text)
                    if text_len > LARGE_TEXT_THRESHOLD:
                        part.text = (
                            part.text[:LARGE_TEXT_KEEP]
                            + f"\n\n[... truncated {text_len - LARGE_TEXT_KEEP} chars ...]"
                        )


async def _create_event_summary_with_llm(events, model_name, session_service, session) -> str:
    event_descriptions = []
    for i, event in enumerate(events):
        author = event.author or "unknown"
        content_texts = []
        tool_calls = []
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    content_texts.append(part.text[:500])
                if hasattr(part, 'function_call') and part.function_call:
                    tool_calls.append(part.function_call.name)
        desc_parts = [f"Event {i} [{author}]"]
        if tool_calls:
            desc_parts.append(f"Tools: {', '.join(tool_calls)}")
        if content_texts:
            desc_parts.append(f"Content: {' | '.join(content_texts)}")
        event_descriptions.append(" - ".join(desc_parts))

    events_text = "\n".join(event_descriptions[:100])
    prompt = (
        f"Summarize these {len(events)} agent events concisely (200-400 words).\n\n"
        f"Events:\n{events_text}\n\n"
        "Cover: agents involved, major actions, files created, outcomes, errors, and current state."
    )

    try:
        llm = LiteLlm(
            model=model_name,
            num_retries=3,
            timeout=60,
            api_base=OLLAMA_BASE_URL,
        )
        llm_request = LlmRequest(
            model=model_name,
            contents=[genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])],
            config=genai_types.GenerateContentConfig(temperature=0.3, max_output_tokens=2000),
        )
        response = None
        async for llm_response in llm.generate_content_async(llm_request=llm_request, stream=False):
            response = llm_response
            break

        if response and response.content and response.content.parts:
            parts = response.content.parts
            if parts and parts[0].text:
                return parts[0].text

        return f"[COMPRESSED] {len(events)} events from agents: {', '.join(set(e.author for e in events if e.author))}"

    except Exception as e:
        logger.warning(f"[Compression] LLM summarization failed: {e}")
        return f"[COMPRESSED] {len(events)} events. Summary unavailable."


async def _compress_session_events(session, start_idx, end_idx, summary_text, session_service):
    events = session.events
    if start_idx >= end_idx or end_idx > len(events):
        return

    start_timestamp = events[start_idx].timestamp if start_idx < len(events) else 0.0
    end_timestamp = events[end_idx - 1].timestamp if end_idx <= len(events) else 0.0

    compaction = EventCompaction(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        compacted_content=genai_types.Content(
            role='model', parts=[genai_types.Part(text=summary_text)]
        ),
    )
    summary_event = Event(
        author='event_compression',
        actions=EventActions(compaction=compaction),
        invocation_id=events[0].invocation_id if events else None,
    )
    new_events = events[:start_idx] + [summary_event] + events[end_idx:]
    _truncate_large_event_texts(new_events)
    session.events = new_events
    logger.info(f"[Compression] Compressed {end_idx - start_idx} events → 1 summary event")


def create_compression_callback(
    event_threshold: int = DEFAULT_EVENT_THRESHOLD,
    overlap_size: int = DEFAULT_OVERLAP_SIZE,
    model_name: Optional[str] = None,
):
    """Create an event compression callback that uses local Ollama for summarization."""
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME

    async def compression_callback(callback_context: CallbackContext):
        ctx = callback_context._invocation_context
        session = ctx.session
        events = session.events
        event_count = len(events)

        if event_count <= event_threshold:
            return None

        logger.info(f"[Compression] {event_count} events exceeds threshold {event_threshold}, compressing...")

        end_idx = event_count - overlap_size
        if end_idx <= 0:
            return None

        last_compaction_idx = -1
        for i in range(len(events) - 1, -1, -1):
            if events[i].actions and events[i].actions.compaction:
                last_compaction_idx = i
                break

        start_idx = max(0, last_compaction_idx + 1)
        if start_idx >= end_idx:
            return None

        events_to_compress = events[start_idx:end_idx]
        _truncate_large_event_texts(events_to_compress)

        try:
            summary_text = await _create_event_summary_with_llm(
                events=events_to_compress,
                model_name=model_name,
                session_service=ctx.session_service,
                session=session,
            )
            await _compress_session_events(
                session=session,
                start_idx=start_idx,
                end_idx=end_idx,
                summary_text=summary_text,
                session_service=ctx.session_service,
            )
        except Exception as e:
            logger.error(f"[Compression] Failed: {e}", exc_info=True)

        return None

    return compression_callback


async def compress_events_manually(ctx, event_threshold=DEFAULT_EVENT_THRESHOLD,
                                    overlap_size=DEFAULT_OVERLAP_SIZE, model_name=None):
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME

    session = ctx.session
    events = session.events
    event_count = len(events)

    if event_count <= event_threshold:
        return

    end_idx = event_count - overlap_size
    if end_idx <= 0:
        return

    last_compaction_idx = -1
    for i in range(len(events) - 1, -1, -1):
        if events[i].actions and events[i].actions.compaction:
            last_compaction_idx = i
            break

    start_idx = max(0, last_compaction_idx + 1)
    if start_idx >= end_idx:
        return

    events_to_compress = events[start_idx:end_idx]
    _truncate_large_event_texts(events_to_compress)

    try:
        summary_text = await _create_event_summary_with_llm(
            events=events_to_compress,
            model_name=model_name,
            session_service=ctx.session_service,
            session=session,
        )
        await _compress_session_events(
            session=session,
            start_idx=start_idx,
            end_idx=end_idx,
            summary_text=summary_text,
            session_service=ctx.session_service,
        )
    except Exception as e:
        logger.error(f"[ManualCompression] Failed: {e}", exc_info=True)


def create_hard_limit_callback(max_events: int = 50):
    def hard_limit_callback(callback_context: CallbackContext):
        session = callback_context._invocation_context.session
        events = session.events
        if len(events) > max_events:
            session.events = events[-max_events:]
            logger.info(f"[HardLimit] Trimmed to {max_events} events")
        return None
    return hard_limit_callback
