"""
Research — Standalone MCP server for agentic web and X platform research.

Wraps Grok's agentic capabilities behind a simple tool interface.
Returns results inline when fast enough, otherwise hands back a task_id
for async retrieval.

Environment:
  GROK_API_URL        Grok API base URL (default: https://api.x.ai/v1)
  GROK_API_KEY        Grok API key (required)
  RESEARCH_TIMEOUT    Max seconds to wait for inline results (default: 30)
"""
from fastmcp import FastMCP
from typing import Optional, Annotated
from pydantic import Field
from openai import AsyncOpenAI
import asyncio
import time
import uuid
import re
import os


# ─── Configuration ──────────────────────────────────────

GROK_API_URL = os.environ.get("GROK_API_URL", "https://api.x.ai/v1")
GROK_API_KEY = os.environ.get("GROK_API_KEY")
RESEARCH_TIMEOUT = int(os.environ.get("RESEARCH_TIMEOUT", "30"))

if not GROK_API_KEY:
    raise RuntimeError("GROK_API_KEY environment variable is required")

grok_client = AsyncOpenAI(api_key=GROK_API_KEY, base_url=GROK_API_URL)


# ─── Task Storage ───────────────────────────────────────

_tasks = {}
_MAX_TASKS = 100
_TASK_EXPIRY_HOURS = 24


def _cleanup_tasks():
    """Remove expired or excess tasks."""
    cutoff = time.time() - (_TASK_EXPIRY_HOURS * 3600)
    expired = [tid for tid, t in _tasks.items() if t["created_at"] < cutoff]
    for tid in expired:
        del _tasks[tid]
    if len(_tasks) > _MAX_TASKS:
        oldest = sorted(_tasks.items(), key=lambda x: x[1]["created_at"])
        for tid, _ in oldest[: len(_tasks) - _MAX_TASKS]:
            del _tasks[tid]


async def _run_grok(task_id: str, prompt: str):
    """Execute research in background and store results."""
    try:
        _tasks[task_id]["status"] = "running"
        stream = await grok_client.chat.completions.create(
            model="grok-4.20-fast",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            stream=True,
        )
        result = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content
        result = re.sub(
            r'<think>.*?</think>', '', result, flags=re.DOTALL
        ).strip()
        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["result"] = result
    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)
    finally:
        _tasks[task_id]["event"].set()


# ─── MCP Server ─────────────────────────────────────────

mcp = FastMCP("Research")


@mcp.tool
async def research(
    prompt: Annotated[
        Optional[str],
        Field(
            description=(
                "Research request. Provide objective, context, key "
                "questions, and desired format. Detailed prompts "
                "produce better results."
            )
        ),
    ] = None,
    task_id: Annotated[
        Optional[str],
        Field(
            description=(
                "Task ID from a previous research call to retrieve results."
            )
        ),
    ] = None,
) -> str:
    """Agentic Web and X platform research via Claude. Returns results \
directly when fast enough — otherwise hands back a task_id \
to check later. Prompt quality determines output quality — \
be thorough about objectives and format."""
    if task_id:
        _cleanup_tasks()
        if task_id not in _tasks:
            return f"Task '{task_id}' not found or expired."
        task = _tasks[task_id]
        status = task["status"]
        if status in ("pending", "running"):
            elapsed = int(time.time() - task["created_at"])
            return (
                f"Still running ({elapsed}s elapsed). "
                f"Check again shortly."
            )
        elif status == "completed":
            return task["result"]
        elif status == "failed":
            return f"Research failed: {task['error']}"
        return f"Unknown task state: {status}"
    elif prompt:
        _cleanup_tasks()
        tid = f"research_{uuid.uuid4().hex[:8]}"
        event = asyncio.Event()
        _tasks[tid] = {
            "status": "pending",
            "created_at": time.time(),
            "result": None,
            "error": None,
            "event": event,
        }
        asyncio.create_task(_run_grok(tid, prompt))
        try:
            await asyncio.wait_for(event.wait(), timeout=RESEARCH_TIMEOUT)
        except asyncio.TimeoutError:
            return (
                f"Research started: {tid}\n"
                f"Taking longer than expected.\n"
                f"Call research(task_id='{tid}') to get results when ready."
            )
        if _tasks[tid]["status"] == "completed":
            return _tasks[tid]["result"]
        return f"Research failed: {_tasks[tid]['error']}"
    else:
        return (
            "Provide either a prompt to start research "
            "or a task_id to check results."
        )


def main():
    mcp.run()


if __name__ == "__main__":
    main()
