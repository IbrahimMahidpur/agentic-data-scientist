# Agentic Data Scientist — Local Ollama Edition 🦙

This is a **fully local** version of the Agentic Data Scientist project.
All cloud API calls have been replaced with your locally downloaded Ollama models.
**No API keys required.**

## Your Models

| Role | Model |
|------|-------|
| Planning, parsing, reflection, summary | `qwen3:8b` |
| Review & confirmation agents | `gemma4:latest` |
| Coding / implementation agent | `gpt-oss:120b-cloud` |

---

## What Was Changed

| File | Change |
|------|--------|
| `agents/adk/utils.py` | Replaced OpenRouter/Google LiteLLM config → Ollama LiteLLM config |
| `agents/adk/agent.py` | Removed `BuiltInPlanner`, `ThinkingConfig`, `ContextCacheConfig` (Google-only features) |
| `agents/adk/review_confirmation.py` | Removed `BuiltInPlanner` / `ThinkingConfig` |
| `agents/adk/implementation_loop.py` | Replaced `ClaudeCodeAgent` → `OllamaCodingAgent` |
| `agents/ollama_coding/` | **NEW** — `OllamaCodingAgent` that calls local Ollama via LiteLLM |
| `core/api.py` | `claude_code` agent type redirected to `adk` with Ollama |
| `pyproject.toml` | Removed `claude-agent-sdk` dependency |
| `.env` | Pre-filled with your Ollama model names |

---

## Step-by-Step Setup in VS Code

### Step 1 — Install Ollama (if not done)
Download from https://ollama.com and install it.

Start the Ollama server:
```bash
ollama serve
```

### Step 2 — Pull your models
Open a terminal and run:
```bash
ollama pull qwen3:8b
ollama pull gemma4:latest
ollama pull gpt-oss:120b-cloud
```

Verify they downloaded:
```bash
ollama list
```

### Step 3 — Open the project in VS Code
```bash
code /path/to/agentic-ds-ollama
```
Or use **File → Open Folder** inside VS Code.

### Step 4 — Install Python 3.12
This project requires **exactly Python 3.12** (not 3.11, not 3.13).

Check your version:
```bash
python3 --version
```

If needed, download from https://www.python.org/downloads/

### Step 5 — Open VS Code Terminal
Press `` Ctrl+` `` (backtick) to open the integrated terminal.

### Step 6 — Create virtual environment
```bash
python3.12 -m venv .venv
```

### Step 7 — Activate virtual environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

### Step 8 — Select Python interpreter in VS Code
1. Press `Ctrl+Shift+P`
2. Type `Python: Select Interpreter`
3. Choose the `.venv` interpreter (should show `3.12.x`)

### Step 9 — Install dependencies
```bash
pip install --upgrade pip
pip install -e ".[dev]"
```
This will take a few minutes.

### Step 10 — Run the quick test
```bash
python examples/quick_test.py
```

Expected output:
```
[PASS] Model response: Hello from Ollama!
[PASS] Agent completed
[ALL TESTS PASSED] Your Ollama setup is working correctly!
```

### Step 11 — Run your first analysis
```bash
python examples/run_analysis.py "Analyze the iris dataset and create visualizations"
```

Or use the CLI:
```bash
agentic-data-scientist run "Your analysis task here"
```

---

## Project Structure

```
agentic-ds-ollama/
├── .env                          ← Ollama model configuration
├── .env.example                  ← Template
├── setup.sh                      ← Automated setup script (macOS/Linux)
├── pyproject.toml                ← Dependencies (claude-agent-sdk removed)
├── examples/
│   ├── quick_test.py             ← Test Ollama connection + agent
│   └── run_analysis.py           ← Example analysis task
├── src/agentic_data_scientist/
│   ├── agents/
│   │   ├── adk/
│   │   │   ├── utils.py          ← ✅ Modified: Ollama LiteLLM config
│   │   │   ├── agent.py          ← ✅ Modified: No ThinkingConfig
│   │   │   ├── review_confirmation.py  ← ✅ Modified
│   │   │   └── implementation_loop.py  ← ✅ Modified: OllamaCodingAgent
│   │   └── ollama_coding/        ← ✅ NEW: OllamaCodingAgent
│   │       ├── __init__.py
│   │       └── agent.py
│   └── core/
│       └── api.py                ← ✅ Modified: claude_code → adk fallback
└── agentic_output/               ← Created at runtime: outputs go here
```

---

## Changing Models

Edit `.env` to swap models:

```env
DEFAULT_MODEL=ollama/qwen3:8b         # or ollama/llama3.2:latest
REVIEW_MODEL=ollama/gemma4:latest      # or ollama/mistral:latest
CODING_MODEL=ollama/gpt-oss:120b-cloud # heaviest model, needs the most RAM
```

Any model in `ollama list` can be used. Prefix with `ollama/`.

---

## Troubleshooting

**"Connection refused" / Ollama not reachable**
```bash
ollama serve   # start in a separate terminal
```

**Model not found**
```bash
ollama pull <model-name>   # download first
```

**Out of memory (OOM)**
Use smaller models. Try:
```env
DEFAULT_MODEL=ollama/qwen3:8b
REVIEW_MODEL=ollama/qwen3:8b
CODING_MODEL=ollama/qwen3:8b
```

**Python version error**
```bash
python3.12 -m venv .venv   # explicitly use 3.12
```

**Permission error on Windows (PowerShell)**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Using with Python API

```python
from agentic_data_scientist import DataScientist

ds = DataScientist(agent_type="adk")
result = ds.run("Analyze sales data and create a report")

print(result.status)         # "completed"
print(result.response)       # full text response
print(result.files_created)  # list of created files
```

## Streaming

```python
import asyncio
from agentic_data_scientist import DataScientist

async def main():
    ds = DataScientist(agent_type="adk")
    async for event in await ds.run_async("Analyze data", stream=True):
        if event["type"] == "message":
            print(event["content"], end="", flush=True)

asyncio.run(main())
```
