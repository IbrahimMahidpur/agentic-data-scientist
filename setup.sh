#!/usr/bin/env bash
# =============================================================
#  setup.sh — One-click setup for Agentic Data Scientist
#             (Ollama / local model edition)
# =============================================================
set -e

echo ""
echo "=================================================="
echo "  Agentic Data Scientist — Ollama Setup Script"
echo "=================================================="
echo ""

# 1. Check Python version
PY_VER=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
PY_MAJOR=$(echo $PY_VER | cut -d. -f1)
PY_MINOR=$(echo $PY_VER | cut -d. -f2)

if [ "$PY_MAJOR" -ne 3 ] || [ "$PY_MINOR" -ne 12 ]; then
    echo "[ERROR] Python 3.12 is required. You have Python $PY_VER"
    echo "        Install Python 3.12: https://www.python.org/downloads/"
    exit 1
fi
echo "[OK] Python $PY_VER detected"

# 2. Check Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo ""
    echo "[WARNING] Ollama is not running or not accessible at http://localhost:11434"
    echo "          Please start Ollama: https://ollama.com"
    echo "          Then pull models:"
    echo "            ollama pull qwen3:8b"
    echo "            ollama pull gemma4:latest"
    echo "            ollama pull gpt-oss:120b-cloud"
    echo ""
    read -p "Continue anyway? (y/N): " CONTINUE
    if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
        exit 1
    fi
else
    echo "[OK] Ollama is running"
    # List available models
    MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; d=json.load(sys.stdin); [print('  -', m['name']) for m in d.get('models',[])]" 2>/dev/null || echo "  (could not list models)")
    echo "[OK] Available Ollama models:"
    echo "$MODELS"
fi

# 3. Create virtual environment
if [ ! -d ".venv" ]; then
    echo ""
    echo "[SETUP] Creating virtual environment (.venv)..."
    python3 -m venv .venv
    echo "[OK] Virtual environment created"
else
    echo "[OK] Virtual environment already exists"
fi

# 4. Activate venv
source .venv/bin/activate
echo "[OK] Virtual environment activated"

# 5. Upgrade pip
pip install --upgrade pip --quiet

# 6. Install the package
echo ""
echo "[SETUP] Installing agentic-data-scientist and dependencies..."
pip install -e ".[dev]" --quiet
echo "[OK] Installation complete"

# 7. Verify .env exists
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[OK] Created .env from .env.example — please review model names"
else
    echo "[OK] .env file found"
fi

echo ""
echo "=================================================="
echo "  Setup complete! To run:"
echo ""
echo "  1. Activate the environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Run a quick test:"
echo "     python examples/quick_test.py"
echo ""
echo "  3. Or use the CLI:"
echo "     agentic-data-scientist run \"Analyze iris dataset\""
echo "=================================================="
