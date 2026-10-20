"""
SENTINEL — AXIOM-OPS Main Agent
Builds and runs the LangChain ReAct agent with SENTINEL monitoring.
"""

from __future__ import annotations

import os
from typing import Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel

import config
from agent.tools   import ALL_TOOLS
from agent.prompts import SYSTEM_PROMPT, SCENARIO_PROMPTS
from agent.session import Session, set_session
from sentinel.interceptor import Interceptor


# ── Prompt template for ReAct ─────────────────────────────────────────────────
# LangChain's ReAct loop requires: {tools}, {tool_names}, {input}, {agent_scratchpad}

_REACT_TEMPLATE = SYSTEM_PROMPT + """

You have access to the following tools:
{tools}

Use the following format EXACTLY:

Thought: [your reasoning about the current step]
Action: [tool name — must be one of: {tool_names}]
Action Input: [input to the tool as a plain string or JSON]
Observation: [result of the tool call]
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information to complete the task.
Final Answer: [your final response to the task]

Begin!

Task: {input}
{agent_scratchpad}"""

_PROMPT = PromptTemplate.from_template(_REACT_TEMPLATE)


def _build_llm() -> BaseLanguageModel:
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
        from langchain_community.llms import Ollama
        return Ollama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE,
            temperature=0,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'openai' or 'ollama'.")


def build_agent(tools=None) -> AgentExecutor:
    """
    Build the AXIOM-OPS AgentExecutor.
    Optionally pass a subset of tools (used by the Least-Privilege Controller).
    """
    llm   = _build_llm()
    tools = tools or ALL_TOOLS

    agent = create_react_agent(llm=llm, tools=tools, prompt=_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=config.AGENT_VERBOSE,
        max_iterations=config.AGENT_MAX_ITERATIONS,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


def run_scenario(scenario_name: str,
                 custom_task: str | None = None,
                 extra_context: dict | None = None) -> dict[str, Any]:
    """
    Run a named scenario (or a custom task) through AXIOM-OPS + SENTINEL.

    Returns a result dict with:
      - output: final agent answer
      - steps:  list of (action, observation) tuples
      - session_summary: dict from Session.summary()
      - events: all intercepted events from the event bus
    """
    # Resolve task text
    task = custom_task or SCENARIO_PROMPTS.get(scenario_name, "")
    if not task:
        raise ValueError(f"Unknown scenario: {scenario_name!r}. "
                         f"Available: {list(SCENARIO_PROMPTS.keys())}")

    # ── Set up fresh session ──────────────────────────────────────────────────
    sess = Session(task=task, scenario=scenario_name)
    set_session(sess)

    # Reset interceptor state for new session
    Interceptor.reset()
    interceptor = Interceptor.get_instance()
    interceptor.intent_analyser.initialise_task(task)

    # Reset event bus — preserve any active dashboard WS subscribers
    from sentinel import event_bus
    existing_subscribers = list(event_bus._subscribers)  # snapshot before clear
    event_bus.clear()
    for sub in existing_subscribers:                      # re-register them
        event_bus.subscribe(sub)

    # Broadcast session start to any connected dashboard
    event_bus.publish({
        "type":       "session_start",
        "scenario":   scenario_name,
        "task":       task,
        "session_id": sess.session_id,
    })

    # ── Print session header ──────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"  SENTINEL | AXIOM-OPS  —  Session {sess.session_id}")
    print(f"  Scenario : {scenario_name}")
    print(f"  Task     : {task[:80]}{'...' if len(task) > 80 else ''}")
    print("═" * 70 + "\n")

    # ── Run agent ─────────────────────────────────────────────────────────────
    executor = build_agent()
    result   = executor.invoke({"input": task})

    # ── Collect outputs ───────────────────────────────────────────────────────
    events  = event_bus.get_events()
    summary = sess.summary()

    # Broadcast session end to any connected dashboard
    event_bus.publish({
        "type":    "session_end",
        "summary": summary,
    })

    _print_session_summary(summary, events)

    return {
        "output":          result.get("output", ""),
        "steps":           result.get("intermediate_steps", []),
        "session_summary": summary,
        "events":          events,
        "session":         sess,
    }


def _print_session_summary(summary: dict, events: list) -> None:
    """Pretty-print a session summary to stdout."""
    print("\n" + "─" * 70)
    print("  SENTINEL SESSION SUMMARY")
    print("─" * 70)
    print(f"  Session ID : {summary['session_id']}")
    print(f"  Scenario   : {summary['scenario']}")
    print(f"  Duration   : {summary['duration_s']}s")
    print(f"  Actions    : {summary['total_actions']}  "
          f"(✓ {summary['allowed']}  ⚠ {summary['warned']}  ✕ {summary['blocked']})")
    print(f"  Violations : {summary['violations']}")
    print(f"  PFA State  : {summary['final_pfa_state']}")
    print(f"  Risk Score : {summary['overall_risk']}")
    print(f"  Monitoring : {summary['monitoring_level']}")

    if events:
        print("\n  Action Log:")
        for ev in events:
            icon = {"ALLOW": "✓", "WARN": "⚠", "BLOCK": "✕"}.get(ev["decision"], "?")
            print(f"    {ev['seq']:>2}. {icon} [{ev['decision']:<5}] "
                  f"{ev['tool']:<22} "
                  f"sec={ev['scores']['security']:.2f} "
                  f"saf={ev['scores']['safety']:.2f} "
                  f"drift={ev['drift']:.2f} "
                  f"| {ev.get('pfa_state','?')}")
            if ev.get("ltl_violations"):
                print(f"         LTL: {', '.join(ev['ltl_violations'])}")
            if ev["decision"] == "BLOCK":
                print(f"         ↳  {ev['reason'][:90]}")
    print("─" * 70 + "\n")
