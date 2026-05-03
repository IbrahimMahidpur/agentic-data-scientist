"""
claude_code agent — DISABLED in Ollama build.
All imports redirect to OllamaCodingAgent.
"""
from agentic_data_scientist.agents.ollama_coding.agent import OllamaCodingAgent as ClaudeCodeAgent

def setup_working_directory(working_dir: str) -> None:
    from pathlib import Path
    for d in ["user_data", "workflow", "results"]:
        (Path(working_dir) / d).mkdir(parents=True, exist_ok=True)

__all__ = ["ClaudeCodeAgent", "setup_working_directory"]
