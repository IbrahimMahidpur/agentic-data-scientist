"""
quick_test.py — Verify the Ollama setup is working end-to-end.

Run:
    python examples/quick_test.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development installs
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def test_ollama_connection():
    """Test that Ollama is reachable and the models respond."""
    import litellm
    from dotenv import load_dotenv
    import os

    load_dotenv()

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    default_model = os.getenv("DEFAULT_MODEL", "ollama/qwen3.5:9b")

    print(f"\n[TEST] Connecting to Ollama at {base_url}")
    print(f"[TEST] Testing model: {default_model}\n")

    try:
        response = await litellm.acompletion(
            model=default_model,
            messages=[{"role": "user", "content": "Say 'Hello from Ollama!' in one sentence."}],
            api_base=base_url,
            max_tokens=50,
            timeout=60,
        )
        msg = response.choices[0].message.content
        print(f"[PASS] Model response: {msg}")
        return True
    except Exception as e:
        print(f"[FAIL] Could not reach Ollama: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running: ollama serve")
        print("  2. Models are pulled: ollama pull qwen3:8b")
        return False


async def test_agent():
    """Test the full agent pipeline with a simple task."""
    from agentic_data_scientist import DataScientist

    print("\n[TEST] Running full agent pipeline...")
    print("[TEST] Task: Print 'Hello World' to a file\n")

    ds = DataScientist(agent_type="adk")
    result = await ds.run_async("Write a Python script that prints 'Hello World' and saves it to hello.txt, then run it.")

    print(f"\n[RESULT] Status: {result.status}")
    if result.response:
        print(f"[RESULT] Response (first 500 chars): {result.response[:500]}")
    if result.files_created:
        print(f"[RESULT] Files created: {result.files_created}")
    if result.error:
        print(f"[ERROR] {result.error}")

    return result.status == "completed"


async def main():
    print("=" * 50)
    print("  Agentic Data Scientist — Ollama Quick Test")
    print("=" * 50)

    ok1 = await test_ollama_connection()
    if not ok1:
        print("\n[ABORT] Ollama connection failed. Fix connection before running agent test.")
        sys.exit(1)

    ok2 = await test_agent()
    if ok2:
        print("\n[ALL TESTS PASSED] Your Ollama setup is working correctly!")
    else:
        print("\n[SOME TESTS FAILED] Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
