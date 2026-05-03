"""
Main ADK agent factory for Agentic Data Scientist — Ollama edition.
ThinkingConfig / BuiltInPlanner removed (not supported by Ollama/LiteLLM).
"""

import logging
import warnings
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from dotenv import load_dotenv
from google.adk.agents import InvocationContext, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.events import Event
from google.adk.utils.context_utils import Aclosing
from google.genai import types
from pydantic import BaseModel, Field
from typing_extensions import override

from agentic_data_scientist.agents.adk.event_compression import create_compression_callback
from agentic_data_scientist.agents.adk.implementation_loop import make_implementation_agents
from agentic_data_scientist.agents.adk.loop_detection import LoopDetectionAgent
from agentic_data_scientist.agents.adk.review_confirmation import create_review_confirmation_agent
from agentic_data_scientist.agents.adk.utils import (
    DEFAULT_MODEL,
    REVIEW_MODEL,
    get_generate_content_config,
    is_network_disabled,
)
from agentic_data_scientist.prompts import load_prompt


load_dotenv()
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="google.adk.tools.mcp_tool")
logging.getLogger("google_genai.types").setLevel(logging.WARNING)


# ========================= Output Schemas =========================

class Stage(BaseModel):
    title: str = Field(description="Stage title")
    description: str = Field(description="Detailed stage description")


class SuccessCriterion(BaseModel):
    criteria: str = Field(description="Success criterion description")


class PlanParserOutput(BaseModel):
    stages: List[Stage] = Field(description="List of high-level stages to implement progressively")
    success_criteria: List[SuccessCriterion] = Field(description="Definitive checklist for overall analysis completion")


class CriteriaUpdate(BaseModel):
    index: int = Field(description="Criterion index")
    met: bool = Field(description="Whether criterion is met")
    evidence: str = Field(description="Evidence or reason for the status")


class CriteriaCheckerOutput(BaseModel):
    criteria_updates: List[CriteriaUpdate] = Field(description="List of criteria with updated met status and evidence")


class StageModification(BaseModel):
    index: int = Field(description="Stage index to modify")
    new_description: str = Field(description="Updated stage description")


class NewStage(BaseModel):
    title: str = Field(description="New stage title")
    description: str = Field(description="New stage description")


class StageReflectorOutput(BaseModel):
    stage_modifications: List[StageModification] = Field(description="Modifications to existing uncompleted stages")
    new_stages: List[NewStage] = Field(description="New stages to add to the end of the stage list")


PLAN_PARSER_OUTPUT_SCHEMA = PlanParserOutput
CRITERIA_CHECKER_OUTPUT_SCHEMA = CriteriaCheckerOutput
STAGE_REFLECTOR_OUTPUT_SCHEMA = StageReflectorOutput


# ========================= Callbacks =========================

def plan_parser_callback(callback_context: CallbackContext):
    ctx = callback_context._invocation_context
    state = ctx.session.state
    parsed_output = state.get("parsed_plan_output")

    if not parsed_output or not isinstance(parsed_output, dict):
        logger.error("[PlanParser] No valid parsed output found in state")
        return

    stages_data = parsed_output.get("stages", [])
    criteria_data = parsed_output.get("success_criteria", [])

    stages = []
    for idx, stage in enumerate(stages_data):
        if not isinstance(stage, dict) or "title" not in stage or "description" not in stage:
            continue
        stages.append({
            "index": idx,
            "title": stage["title"],
            "description": stage["description"],
            "completed": False,
            "implementation_result": None,
        })

    criteria = []
    for idx, crit in enumerate(criteria_data):
        if not isinstance(crit, dict) or "criteria" not in crit:
            continue
        criteria.append({
            "index": idx,
            "criteria": crit["criteria"],
            "met": False,
            "evidence": None,
        })

    if not stages or not criteria:
        logger.error("[PlanParser] No valid stages or criteria after parsing")
        return

    state["high_level_stages"] = stages
    state["high_level_success_criteria"] = criteria
    state["current_stage_index"] = 0
    logger.info(f"[PlanParser] Initialized {len(stages)} stages and {len(criteria)} criteria")


def criteria_checker_callback(callback_context: CallbackContext):
    ctx = callback_context._invocation_context
    state = ctx.session.state
    criteria_output = state.get("criteria_checker_output")
    criteria = state.get("high_level_success_criteria", [])

    if not criteria_output or not isinstance(criteria_output, dict):
        logger.error("[CriteriaChecker] No valid output found in state")
        return

    updates = criteria_output.get("criteria_updates", [])
    for update in updates:
        if not isinstance(update, dict):
            continue
        idx = update.get("index")
        if idx is not None and 0 <= idx < len(criteria):
            criteria[idx]["met"] = update.get("met", False)
            criteria[idx]["evidence"] = update.get("evidence", "")

    met_count = sum(1 for c in criteria if c.get("met", False))
    logger.info(f"[CriteriaChecker] {met_count}/{len(criteria)} criteria met")
    state["high_level_success_criteria"] = criteria


def stage_reflector_callback(callback_context: CallbackContext):
    ctx = callback_context._invocation_context
    state = ctx.session.state
    reflector_output = state.get("stage_reflector_output")
    stages = state.get("high_level_stages", [])

    if not reflector_output or not isinstance(reflector_output, dict):
        logger.error("[StageReflector] No valid output found in state")
        return

    for mod in reflector_output.get("stage_modifications", []):
        if not isinstance(mod, dict):
            continue
        idx = mod.get("index")
        new_desc = mod.get("new_description", "")
        if idx is not None and 0 <= idx < len(stages) and new_desc:
            if not stages[idx].get("completed", False):
                stages[idx]["description"] = new_desc

    for new_stage in reflector_output.get("new_stages", []):
        if not isinstance(new_stage, dict):
            continue
        if "title" not in new_stage or "description" not in new_stage:
            continue
        stages.append({
            "index": len(stages),
            "title": new_stage["title"],
            "description": new_stage["description"],
            "completed": False,
            "implementation_result": None,
        })

    state["high_level_stages"] = stages


class NonEscalatingLoopAgent(LoopAgent):
    """Loop agent that does not propagate escalate flags upward."""

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        times_looped = 0
        while not self.max_iterations or times_looped < self.max_iterations:
            for sub_agent in self.sub_agents:
                should_exit = False
                async with Aclosing(sub_agent.run_async(ctx)) as agen:
                    async for event in agen:
                        if event.actions.escalate:
                            event.actions.escalate = False
                            should_exit = True
                        yield event
                        if should_exit:
                            break
                if should_exit:
                    return
            times_looped += 1


def create_agent(
    working_dir: Optional[str] = None,
    mcp_servers: Optional[List[str]] = None,
) -> LoopDetectionAgent:
    """Factory: create an Agentic Data Scientist ADK agent using Ollama."""
    if working_dir is None:
        import tempfile
        working_dir = tempfile.mkdtemp(prefix="agentic_ds_")

    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[AgenticDS] Creating ADK agent with working_dir={working_dir}")

    from agentic_data_scientist.tools import (
        directory_tree,
        fetch_url,
        get_file_info,
        list_directory,
        read_file,
        read_media_file,
        search_files,
    )

    working_dir_str = str(working_dir)

    def read_file_bound(path: str, head: Optional[int] = None, tail: Optional[int] = None) -> str:
        """Read file contents with optional head/tail line limits."""
        return read_file(path, working_dir_str, head, tail)

    def read_media_file_bound(path: str) -> str:
        """Read binary/media files and return base64 encoded data."""
        return read_media_file(path, working_dir_str)

    def list_directory_bound(path: str = ".", show_sizes: bool = False, sort_by: str = "name") -> str:
        """List directory contents."""
        return list_directory(path, working_dir_str, show_sizes, sort_by)

    def directory_tree_bound(path: str = ".", exclude_patterns: Optional[list] = None) -> str:
        """Generate a recursive directory tree view."""
        return directory_tree(path, working_dir_str, exclude_patterns)

    def search_files_bound(pattern: str, path: str = ".", exclude_patterns: Optional[list] = None) -> str:
        """Search for files matching a pattern."""
        return search_files(pattern, working_dir_str, path, exclude_patterns)

    def get_file_info_bound(path: str) -> str:
        """Get detailed metadata about a file."""
        return get_file_info(path, working_dir_str)

    tools = [
        read_file_bound,
        read_media_file_bound,
        list_directory_bound,
        directory_tree_bound,
        search_files_bound,
        get_file_info_bound,
    ]

    if not is_network_disabled():
        tools.append(fetch_url)

    logger.info(f"[AgenticDS] Configured {len(tools)} local tools")

    # Implementation loop
    coding_agent, review_agent, review_confirmation = make_implementation_agents(str(working_dir), tools)

    implementation_loop = NonEscalatingLoopAgent(
        name="implementation_loop",
        description="Iterative implementation-review-confirmation loop for each stage.",
        sub_agents=[coding_agent, review_agent, review_confirmation],
        max_iterations=5,
    )

    # Summary agent (no ThinkingConfig for Ollama)
    summary_agent_instructions = load_prompt("summary")
    summary_agent = LoopDetectionAgent(
        name="summary_agent",
        model=DEFAULT_MODEL,
        description="Summarizes results into a comprehensive pure text report.",
        instruction=summary_agent_instructions,
        tools=tools,
        generate_content_config=get_generate_content_config(temperature=0.3),
    )

    # Plan maker agent
    plan_maker_instructions = load_prompt("plan_maker")
    plan_maker_compression = create_compression_callback(event_threshold=40, overlap_size=20)
    plan_maker_agent = LoopDetectionAgent(
        name="plan_maker_agent",
        model=DEFAULT_MODEL,
        description="Creates high-level plans for complex tasks.",
        instruction=plan_maker_instructions,
        tools=tools,
        output_key="high_level_plan",
        generate_content_config=get_generate_content_config(temperature=0.6),
        after_agent_callback=plan_maker_compression,
    )

    # Plan reviewer agent
    plan_reviewer_instructions = load_prompt("plan_reviewer")
    plan_reviewer_compression = create_compression_callback(event_threshold=40, overlap_size=20)
    plan_reviewer_agent = LoopDetectionAgent(
        name="plan_reviewer_agent",
        model=REVIEW_MODEL,
        description="Reviews high-level plans for completeness and correctness.",
        instruction=plan_reviewer_instructions,
        tools=tools,
        output_key="plan_review_feedback",
        generate_content_config=get_generate_content_config(temperature=0.3),
        after_agent_callback=plan_reviewer_compression,
    )

    high_level_planning_loop = NonEscalatingLoopAgent(
        name="high_level_planning_loop",
        description="High-level planning through multiple iterations.",
        sub_agents=[
            plan_maker_agent,
            plan_reviewer_agent,
            create_review_confirmation_agent(auto_exit_on_completion=True, prompt_name="plan_review_confirmation"),
        ],
        max_iterations=10,
    )

    # Plan parser
    plan_parser_instructions = load_prompt("plan_parser")
    high_level_plan_parser = LoopDetectionAgent(
        name="high_level_plan_parser",
        model=DEFAULT_MODEL,
        description="Parses high-level plan into stages and success criteria.",
        instruction=plan_parser_instructions,
        tools=[],
        output_schema=PLAN_PARSER_OUTPUT_SCHEMA,
        output_key="parsed_plan_output",
        after_agent_callback=plan_parser_callback,
        generate_content_config=get_generate_content_config(temperature=0.0),
    )

    # Criteria checker
    criteria_checker_instructions = load_prompt("criteria_checker")
    criteria_checker_compression = create_compression_callback(event_threshold=40, overlap_size=20)

    async def combined_criteria_callback(callback_context):
        criteria_checker_callback(callback_context)
        await criteria_checker_compression(callback_context)

    success_criteria_checker = LoopDetectionAgent(
        name="success_criteria_checker",
        model=REVIEW_MODEL,
        description="Checks which high-level success criteria have been met.",
        instruction=criteria_checker_instructions,
        tools=tools,
        output_schema=CRITERIA_CHECKER_OUTPUT_SCHEMA,
        output_key="criteria_checker_output",
        after_agent_callback=combined_criteria_callback,
        generate_content_config=get_generate_content_config(temperature=0.0),
    )

    # Stage reflector
    stage_reflector_instructions = load_prompt("stage_reflector")
    stage_reflector_compression = create_compression_callback(event_threshold=40, overlap_size=20)

    async def combined_reflector_callback(callback_context):
        stage_reflector_callback(callback_context)
        await stage_reflector_compression(callback_context)

    stage_reflector = LoopDetectionAgent(
        name="stage_reflector",
        model=DEFAULT_MODEL,
        description="Reflects on and adapts remaining implementation stages.",
        instruction=stage_reflector_instructions,
        tools=tools,
        output_schema=STAGE_REFLECTOR_OUTPUT_SCHEMA,
        output_key="stage_reflector_output",
        after_agent_callback=combined_reflector_callback,
        generate_content_config=get_generate_content_config(temperature=0.4),
    )

    # Stage orchestrator
    from agentic_data_scientist.agents.adk.stage_orchestrator import StageOrchestratorAgent

    stage_orchestrator = StageOrchestratorAgent(
        implementation_loop=implementation_loop,
        criteria_checker=success_criteria_checker,
        stage_reflector=stage_reflector,
        name="stage_orchestrator",
        description="Orchestrates stage-by-stage implementation with adaptive planning.",
    )

    # Root workflow
    workflow = SequentialAgent(
        name="agentic_data_scientist_workflow",
        description="Complete Agentic Data Scientist workflow with adaptive stage-wise implementation.",
        sub_agents=[
            high_level_planning_loop,
            high_level_plan_parser,
            stage_orchestrator,
            summary_agent,
        ],
    )

    logger.info("[AgenticDS] Agent creation complete")
    return workflow


def create_app(
    working_dir: Optional[str] = None,
    mcp_servers: Optional[List[str]] = None,
) -> App:
    """Create an App instance for the ADK agent (Ollama edition, no context caching)."""
    root_agent = create_agent(working_dir=working_dir, mcp_servers=mcp_servers)

    # ContextCacheConfig is a Google-specific feature; skip it for Ollama
    app = App(
        name="agentic-data-scientist",
        root_agent=root_agent,
    )

    logger.info("[AgenticDS] Created App (Ollama mode, context caching disabled)")
    return app
