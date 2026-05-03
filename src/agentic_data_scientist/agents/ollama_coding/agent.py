"""
OllamaCodingAgent — replaces ClaudeCodeAgent for local Ollama inference.

Uses LiteLLM's direct Ollama API to execute coding tasks.
The agent is an ADK Agent subclass that:
  1. Reads the current stage / task from session state
  2. Calls the local Ollama coding model via LiteLLM
  3. Streams output back as ADK Events
  4. Can execute Python code it writes via subprocess to verify results
"""

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, List, Optional

from dotenv import load_dotenv
from google.adk.agents import Agent, InvocationContext
from google.adk.events import Event
from google.genai import types

from agentic_data_scientist.agents.adk.utils import CODING_MODEL_NAME, OLLAMA_BASE_URL


load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert data scientist and software engineer working in a local workspace.
Your job is to implement the given task step by step.

Guidelines:
- Write clean, well-documented Python code
- Save all outputs (plots, CSVs, reports) to the working directory
- When writing code, wrap executable blocks in ```python ... ``` fences
- After writing code, run it using the bash_exec tool to verify it works
- If code fails, debug and fix it
- Summarize what you accomplished at the end
- Keep responses focused and avoid unnecessary verbosity

Working directory: {working_dir}
"""


def _run_python_code(code: str, working_dir: str, timeout: int = 120) -> tuple[bool, str]:
    """
    Execute Python code in the working directory.
    Returns (success, output_text).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=working_dir, delete=False) as f:
        f.write(code)
        script_path = f.name

    try:
        result = subprocess.run(
            ["python", script_path],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]:\n{result.stderr}"
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"Code execution timed out after {timeout}s"
    except Exception as e:
        return False, f"Execution error: {e}"
    finally:
        try:
            os.unlink(script_path)
        except Exception:
            pass


class OllamaCodingAgent(Agent):
    """
    ADK Agent that uses a local Ollama model for coding tasks.
    Replaces ClaudeCodeAgent when no Anthropic API key is available.
    """

    model_config = {"extra": "allow"}

    _working_dir: Optional[str] = None
    _output_key: str = "implementation_summary"
    _tools_list: List = []

    def __init__(
        self,
        name: str = "ollama_coding_agent",
        description: Optional[str] = None,
        working_dir: Optional[str] = None,
        output_key: str = "implementation_summary",
        tools: Optional[List] = None,
        after_agent_callback: Optional[Any] = None,
        **kwargs: Any,
    ):
        model_name = CODING_MODEL_NAME  # e.g. "ollama/gpt-oss:120b-cloud"
        super().__init__(
            name=name,
            description=description or "Coding agent using local Ollama model",
            model=model_name,
            after_agent_callback=after_agent_callback,
            **kwargs,
        )
        self._working_dir = working_dir
        self._output_key = output_key
        self._tools_list = tools or []

    @property
    def working_dir(self) -> Optional[str]:
        return self._working_dir

    @property
    def output_key(self) -> str:
        return self._output_key

    def _setup_working_dir(self, working_dir: str) -> None:
        """Create required directory structure."""
        working_path = Path(working_dir)
        for subdir in ["user_data", "workflow", "results"]:
            (working_path / subdir).mkdir(parents=True, exist_ok=True)

        readme = working_path / "README.md"
        if not readme.exists():
            readme.write_text(
                f"# Agentic Data Scientist Session\n\nWorking Directory: `{working_dir}`\n\n"
                "## Structure\n- `user_data/` — Input files\n- `workflow/` — Scripts\n- `results/` — Outputs\n"
            )

    async def _call_ollama(self, messages: list, stream: bool = True) -> AsyncGenerator[str, None]:
        """
        Call Ollama via LiteLLM's async API and yield text chunks.
        Uses the OpenAI-compatible endpoint at OLLAMA_BASE_URL.
        """
        try:
            import litellm

            # Build the model string for LiteLLM
            model = CODING_MODEL_NAME  # "ollama/qwen3:8b" etc.

            response = await litellm.acompletion(
                model=model,
                messages=messages,
                api_base=OLLAMA_BASE_URL,
                stream=True,
                temperature=0.2,
                timeout=300,
            )

            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

        except Exception as e:
            logger.error(f"[OllamaCodingAgent] LiteLLM call failed: {e}")
            yield f"\n[ERROR calling Ollama: {e}]\n"

    def _extract_python_blocks(self, text: str) -> list[str]:
        """Extract ```python ... ``` code blocks from model output."""
        blocks = []
        in_block = False
        current = []
        for line in text.split("\n"):
            if line.strip().startswith("```python"):
                in_block = True
                current = []
            elif line.strip() == "```" and in_block:
                in_block = False
                blocks.append("\n".join(current))
            elif in_block:
                current.append(line)
        return blocks

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Main agent execution loop."""
        try:
            working_dir = self._working_dir
            if not working_dir:
                working_dir = tempfile.mkdtemp(prefix="ollama_ds_")

            self._setup_working_dir(working_dir)
            state = ctx.session.state

            # Determine the task
            current_stage = state.get("current_stage")
            if current_stage:
                task = (
                    f"Stage {current_stage.get('index', 0) + 1}: {current_stage.get('title', 'Task')}\n\n"
                    f"{current_stage.get('description', '')}"
                )
            else:
                task = (
                    state.get("implementation_task")
                    or state.get("original_user_input")
                    or state.get("latest_user_input")
                    or "No task specified."
                )

            # List any user data files
            user_data_dir = Path(working_dir) / "user_data"
            user_files = list(user_data_dir.glob("*")) if user_data_dir.exists() else []
            files_note = ""
            if user_files:
                files_note = "\n\nAvailable input files in user_data/:\n" + "\n".join(
                    f"  - {f.name} ({f.stat().st_size // 1024} KB)" for f in user_files if f.is_file()
                )

            system = SYSTEM_PROMPT.format(working_dir=working_dir)
            user_message = f"{task}{files_note}\n\nPlease implement this step by step and save all outputs."

            # Yield start event
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=f"[OllamaCodingAgent] Starting with model: {CODING_MODEL_NAME}")],
                ),
            )

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ]

            # Stream response from Ollama
            full_response = ""
            chunk_buffer = ""

            async for chunk in self._call_ollama(messages):
                full_response += chunk
                chunk_buffer += chunk

                # Yield buffered chunks every ~200 chars for smooth streaming
                if len(chunk_buffer) >= 200:
                    yield Event(
                        author=self.name,
                        content=types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=chunk_buffer)],
                        ),
                    )
                    chunk_buffer = ""

            # Flush remaining buffer
            if chunk_buffer:
                yield Event(
                    author=self.name,
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=chunk_buffer)],
                    ),
                )

            # Execute any Python code blocks found in the response
            code_blocks = self._extract_python_blocks(full_response)
            execution_results = []

            for i, code in enumerate(code_blocks, 1):
                yield Event(
                    author=self.name,
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=f"\n[Executing code block {i}/{len(code_blocks)}...]")],
                    ),
                )

                success, output = await asyncio.to_thread(_run_python_code, code, working_dir)
                status = "✅ Success" if success else "❌ Failed"
                exec_msg = f"\n[Code block {i} — {status}]\n{output[:2000]}"
                execution_results.append(exec_msg)

                yield Event(
                    author=self.name,
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=exec_msg)],
                    ),
                )

                # If code failed, ask model to fix it
                if not success and i == len(code_blocks):
                    fix_messages = messages + [
                        {"role": "assistant", "content": full_response},
                        {"role": "user", "content": f"The code failed with:\n{output}\n\nPlease fix and provide the corrected code."},
                    ]
                    fix_response = ""
                    async for chunk in self._call_ollama(fix_messages):
                        fix_response += chunk

                    if fix_response:
                        yield Event(
                            author=self.name,
                            content=types.Content(
                                role="model",
                                parts=[types.Part.from_text(text=f"\n[Fix attempt]:\n{fix_response}")],
                            ),
                        )
                        full_response += f"\n\n[Fix attempt]:\n{fix_response}"

                        # Try executing fixed code
                        fix_blocks = self._extract_python_blocks(fix_response)
                        for fix_code in fix_blocks:
                            success2, output2 = await asyncio.to_thread(_run_python_code, fix_code, working_dir)
                            status2 = "✅ Fixed" if success2 else "❌ Still failing"
                            fix_exec = f"\n[Fix execution — {status2}]\n{output2[:1000]}"
                            yield Event(
                                author=self.name,
                                content=types.Content(
                                    role="model",
                                    parts=[types.Part.from_text(text=fix_exec)],
                                ),
                            )

            # Build summary
            summary = full_response
            if execution_results:
                summary += "\n\n=== Execution Results ===\n" + "\n".join(execution_results)

            # Truncate if needed
            MAX_CHARS = 40000
            if len(summary) > MAX_CHARS:
                summary = summary[:MAX_CHARS * 3 // 4] + "\n\n[...truncated...]\n\n" + summary[-MAX_CHARS // 4:]

            state[self._output_key] = summary

            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="\n=== Task Completed ===")],
                ),
            )

        except Exception as e:
            logger.error(f"[OllamaCodingAgent] Error: {e}", exc_info=True)
            state = ctx.session.state
            state[self._output_key] = f"Error: {e}"
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=f"[OllamaCodingAgent Error]: {e}")],
                ),
            )
