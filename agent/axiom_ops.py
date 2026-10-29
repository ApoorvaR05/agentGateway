"""
SENTINEL — AXIOM-OPS Main Agent
Builds and runs the LangChain agent with SENTINEL monitoring.
Uses the LangChain 1.x create_agent API (backed by LangGraph).
"""

from __future__ import annotations

import time
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.callbacks import BaseCallbackHandler

import config
from agent.tools   import ALL_TOOLS
from agent.prompts import SYSTEM_PROMPT, SCENARIO_PROMPTS
from agent.session import Session, SessionCancelled, set_session
from sentinel.interceptor import Interceptor

try:
    from db import store as db_store
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False


def _build_llm() -> BaseChatModel:
    """Build the LLM based on configuration."""
    provider = config.LLM_PROVIDER

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0,
            api_key=config.OPENAI_API_KEY,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE,
            temperature=0,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'openai' or 'ollama'.")


# ── Callback handler — broadcasts agent thoughts to the event bus ─────────────

class _SentinelCallbackHandler(BaseCallbackHandler):
    """Streams AXIOM-OPS reasoning steps to the event bus so the Agent page
    can display them in real time alongside SENTINEL decisions."""

    def _publish_thought(self, text: str) -> None:
        if not text.strip():
            return
        try:
            from sentinel import event_bus
            event_bus.publish({
                "type":    "agent_thought",
                "content": text.strip()[:600],
                "ts":      time.time(),
            })
        except Exception:
            pass

    def on_llm_end(self, response, **kwargs):
        """Fires for plain LLM models."""
        try:
            if response.generations:
                gen  = response.generations[0][0]
                text = getattr(gen, "text", "")
                self._publish_thought(text)
        except Exception:
            pass

    def on_chat_model_end(self, response, **kwargs):
        """Fires for ChatModels (ChatOllama, ChatOpenAI) — this is the right hook."""
        try:
            if response.generations:
                gen = response.generations[0][0]
                # ChatGeneration stores content in .message.content
                msg  = getattr(gen, "message", None)
                text = getattr(msg, "content", "") if msg else getattr(gen, "text", "")
                self._publish_thought(str(text))
        except Exception:
            pass

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Fires just before a tool is called — before SENTINEL intercepts it."""
        try:
            from sentinel import event_bus
            event_bus.publish({
                "type":      "agent_tool_start",
                "tool":      serialized.get("name", "unknown"),
                "input":     str(input_str)[:300],
                "ts":        time.time(),
            })
        except Exception:
            pass

    def on_tool_end(self, output, **kwargs):
        """Fires after a tool returns its result (post-SENTINEL)."""
        try:
            from sentinel import event_bus
            event_bus.publish({
                "type":   "agent_tool_end",
                "output": str(output)[:400],
                "ts":     time.time(),
            })
        except Exception:
            pass


def build_agent(tools=None):
    """
    Build the AXIOM-OPS agent using LangChain 1.x create_agent.
    Returns a compiled LangGraph runnable.
    """
    llm   = _build_llm()
    tools = tools or ALL_TOOLS
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )


def run_scenario(scenario_name: str,
                 custom_task: str | None = None,
                 extra_context: dict | None = None) -> dict[str, Any]:
    """
    Run a named scenario (or a custom task) through AXIOM-OPS + SENTINEL.

    Returns a result dict with:
      - output:           final agent answer (string)
      - steps:            list of (tool_name, args, result) triples
      - session_summary:  dict from Session.summary()
      - events:           all intercepted events from the event bus
      - session:          the Session object
    """
    # Resolve task text
    task = custom_task or SCENARIO_PROMPTS.get(scenario_name, "")
    if not task:
        raise ValueError(f"Unknown scenario: {scenario_name!r}. "
                         f"Available: {list(SCENARIO_PROMPTS.keys())}")

    # ── Set up fresh session ──────────────────────────────────────────────────
    sess = Session(task=task, scenario=scenario_name)
    sess.set_status("planning")
    set_session(sess)

    # Persist task record
    if _DB_AVAILABLE:
        try:
            db_store.upsert_task(sess.task_id, task, scenario_name,
                                 status="planning")
        except Exception:
            pass

    # Reset interceptor state for new session
    Interceptor.reset()
    interceptor = Interceptor.get_instance()
    interceptor.intent_analyser.initialise_task(task)

    # Reset event bus ring buffer — keep all subscribers intact
    from sentinel import event_bus
    event_bus.clear_events_only()

    # Broadcast session start to any connected dashboard
    event_bus.publish({
        "type":       "session_start",
        "scenario":   scenario_name,
        "task":       task,
        "session_id": sess.session_id,
        "task_id":    sess.task_id,
    })

    # ── Print session header ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  SENTINEL | AXIOM-OPS  --  Session {sess.session_id}")
    print(f"  Scenario : {scenario_name}")
    print(f"  Task     : {task[:80]}{'...' if len(task) > 80 else ''}")
    print(f"  LLM      : {config.LLM_PROVIDER} / "
          f"{config.OLLAMA_MODEL if config.LLM_PROVIDER == 'ollama' else config.OPENAI_MODEL}")
    print("=" * 70 + "\n")

    # ── Run agent ─────────────────────────────────────────────────────────────
    sess.set_status("running")
    if _DB_AVAILABLE:
        try:
            db_store.update_task_status(sess.task_id, "running")
        except Exception:
            pass

    agent    = build_agent()
    callback = _SentinelCallbackHandler()
    try:
        result   = agent.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={"callbacks": [callback]},
        )
        messages = result.get("messages", [])
    except Exception as exc:
        cancelled = sess.cancel_event.is_set()
        sess.set_status("cancelled" if cancelled else "failed")
        if _DB_AVAILABLE:
            try:
                db_store.update_task_status(sess.task_id, sess.status,
                                            summary=sess.summary())
            except Exception:
                pass
        if cancelled:
            raise SessionCancelled("Agent run stopped by user.") from exc
        raise

    # ── Extract final answer and intermediate steps ───────────────────────────
    output = ""
    steps  = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            output = str(msg.content)          # last AI text = final answer
        # Capture tool calls from AIMessage tool_calls attribute
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                steps.append((tc.get("name", ""), tc.get("args", {}), ""))
    # Backfill tool results into steps
    tool_results = [m for m in messages if isinstance(m, ToolMessage)]
    for i, tr in enumerate(tool_results):
        if i < len(steps):
            steps[i] = (steps[i][0], steps[i][1], str(tr.content)[:200])

    # ── Collect outputs ───────────────────────────────────────────────────────
    sess.set_status("verifying")
    events  = event_bus.get_events()
    summary = sess.summary()

    # Determine final status
    final_status = "cancelled" if sess.cancel_event.is_set() else "completed"
    if final_status != "cancelled" and sess.blocked_count > 0 and sess.allowed_count == 0:
        final_status = "blocked"
    sess.set_status(final_status)
    summary["status"] = final_status

    # Persist final task state
    if _DB_AVAILABLE:
        try:
            db_store.update_task_status(sess.task_id, final_status,
                                        summary=summary)
        except Exception:
            pass

    # Broadcast session end to any connected dashboard
    event_bus.publish({
        "type":    "session_end",
        "summary": summary,
        "output":  output,
        "task_id": sess.task_id,
    })

    _print_session_summary(summary, events)

    return {
        "output":          output,
        "steps":           steps,
        "session_summary": summary,
        "events":          events,
        "session":         sess,
    }


def _print_session_summary(summary: dict, events: list) -> None:
    """Pretty-print a session summary to stdout."""
    print("\n" + "-" * 70)
    print("  SENTINEL SESSION SUMMARY")
    print("-" * 70)
    print(f"  Session ID : {summary['session_id']}")
    print(f"  Scenario   : {summary['scenario']}")
    print(f"  Duration   : {summary['duration_s']}s")
    print(f"  Actions    : {summary['total_actions']}  "
          f"(OK {summary['allowed']}  WARN {summary['warned']}  BLOCK {summary['blocked']})")
    print(f"  Violations : {summary['violations']}")
    print(f"  PFA State  : {summary['final_pfa_state']}")
    print(f"  Risk Score : {summary['overall_risk']}")
    print(f"  Monitoring : {summary['monitoring_level']}")

    if events:
        print("\n  Action Log:")
        for ev in events:
            if "decision" not in ev:
                continue
            icon = {"ALLOW": "[OK]", "WARN": "[WARN]", "BLOCK": "[BLOCK]"}.get(ev["decision"], "[?]")
            print(f"    {ev['seq']:>2}. {icon:<7} "
                  f"{ev['tool']:<22} "
                  f"sec={ev['scores']['security']:.2f} "
                  f"saf={ev['scores']['safety']:.2f} "
                  f"drift={ev['drift']:.2f} "
                  f"| {ev.get('pfa_state','?')}")
            if ev.get("ltl_violations"):
                print(f"         LTL: {', '.join(ev['ltl_violations'])}")
            if ev["decision"] == "BLOCK":
                print(f"         ->  {ev['reason'][:90]}")
    print("-" * 70 + "\n")
