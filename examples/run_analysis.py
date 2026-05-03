"""
run_analysis.py — Example: run a data analysis task.

Usage:
    python examples/run_analysis.py "Analyze the iris dataset and create visualizations"
    python examples/run_analysis.py "Build a machine learning model to classify flowers"
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main():
    from agentic_data_scientist import DataScientist

    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Load the iris dataset from sklearn, perform exploratory data analysis, "
        "create visualization plots, and build a simple classification model. "
        "Save all outputs to the results/ folder."
    )

    print(f"\nTask: {task}\n")
    print("=" * 60)

    ds = DataScientist(agent_type="adk", working_dir="./agentic_output")

    async for event in await ds.run_async(task, stream=True):
        event_type = event.get("type", "")

        if event_type == "message":
            author = event.get("author", "agent")
            content = event.get("content", "")
            if content and not event.get("is_thought"):
                print(f"[{author}] {content}", end="", flush=True)

        elif event_type == "function_call":
            name = event.get("name", "")
            print(f"\n  [TOOL] → {name}(...)", flush=True)

        elif event_type == "completed":
            files = event.get("files_created", [])
            duration = event.get("duration", 0)
            print(f"\n\n{'=' * 60}")
            print(f"✅ Completed in {duration:.1f}s")
            if files:
                print(f"📁 Files created ({len(files)}):")
                for f in files[:10]:
                    print(f"   - {f}")
            print(f"{'=' * 60}\n")

        elif event_type == "error":
            print(f"\n[ERROR] {event.get('content', 'Unknown error')}\n")


if __name__ == "__main__":
    asyncio.run(main())
