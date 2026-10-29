"""
SENTINEL — Main Entry Point

Usage
-----
# Run a single named scenario:
    python main.py --scenario normal_deploy
    python main.py --scenario prompt_injection
    python main.py --scenario privilege_escalation
    python main.py --scenario goal_drift
    python main.py --scenario emergent_harm

# Run all 5 scenarios:
    python main.py --all

# Run a custom task:
    python main.py --task "Deploy the worker service to v1.5.0"

# List available scenarios:
    python main.py --list

Environment variables:
    LLM_PROVIDER    openai (default) | ollama
    OPENAI_API_KEY  required when LLM_PROVIDER=openai
    OPENAI_MODEL    gpt-4o (default)
    OLLAMA_MODEL    llama3 (default)
"""

import argparse
import sys

from agent.axiom_ops  import run_scenario
from agent.scenarios  import run_all_scenarios

AVAILABLE_SCENARIOS = [
    "normal_deploy",
    "health_monitor",
    "intent_drift_demo",
    "prompt_injection",
    "privilege_escalation",
    "goal_drift",
    "emergent_harm",
]


def main():
    parser = argparse.ArgumentParser(
        description="SENTINEL — AXIOM-OPS agent runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--scenario", "-s",  choices=AVAILABLE_SCENARIOS,
                        help="Run a named scenario")
    parser.add_argument("--all",      "-a",  action="store_true",
                        help="Run all 5 demo scenarios")
    parser.add_argument("--task",     "-t",  type=str,
                        help="Run a custom task string")
    parser.add_argument("--list",     "-l",  action="store_true",
                        help="List available scenarios and exit")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable scenarios:")
        for name in AVAILABLE_SCENARIOS:
            print(f"  {name}")
        return

    if args.all:
        run_all_scenarios()
        return

    if args.scenario:
        run_scenario(args.scenario)
        return

    if args.task:
        run_scenario("custom", custom_task=args.task)
        return

    # Default: run the normal deploy as a smoke test
    print("No argument given — running Scenario 1 (normal_deploy) as default.\n"
          "Use --help to see all options.")
    run_scenario("normal_deploy")


if __name__ == "__main__":
    main()
