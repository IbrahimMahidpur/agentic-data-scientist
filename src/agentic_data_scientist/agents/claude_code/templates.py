"""Stub — not used in Ollama build."""

def get_claude_context(**kwargs) -> str:
    return str(kwargs)

def get_claude_instructions(**kwargs) -> str:
    return ""

def get_minimal_pyproject() -> str:
    return '[project]\nname = "workspace"\nversion = "0.1.0"\n'
