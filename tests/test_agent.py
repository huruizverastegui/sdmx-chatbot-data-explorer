"""
End-to-end agent tests. Runs the full pipeline with a mock selector that
always picks the first indicator returned by search_data_dictionary.

Run with:
  conda run -n genai_vector python -m pytest tests/test_agent.py -v
"""

import json
import os
import time
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

from agents.sdmx_agent import (
    SYSTEM_PROMPT, TOOLS, _TOOL_MAP,
    reset_session_data, get_fetched_dfs, get_viz_spec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_select(question: str) -> str:
    """Mock selector — always picks the first option."""
    return "User selected: 1"


def run_agent(user_question: str, timeout_s: int = 120) -> dict:
    """
    Run the full agent loop, auto-selecting the first indicator.
    Returns a result dict with keys: answer, fetched_dfs, viz_spec, tool_calls.
    """
    reset_session_data()

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_LLM_API_VERSION"],
    )
    deployment = os.environ["AZURE_OPENAI_LLM_DEPLOYMENT"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    tool_call_log = []
    deadline = time.time() + timeout_s

    with patch.dict(_TOOL_MAP, {"ask_user_to_select_indicator": _auto_select}):
        while time.time() < deadline:
            response = client.chat.completions.create(
                model=deployment, messages=messages, tools=TOOLS, tool_choice="auto",
            )
            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                return {
                    "answer": msg.content,
                    "fetched_dfs": get_fetched_dfs(),
                    "viz_spec": get_viz_spec(),
                    "tool_calls": tool_call_log,
                }

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                tool_map = dict(_TOOL_MAP, ask_user_to_select_indicator=_auto_select)
                result = tool_map[fn_name](**fn_args)
                tool_call_log.append({"tool": fn_name, "args": fn_args, "result_snippet": str(result)[:200]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    raise TimeoutError(f"Agent did not finish within {timeout_s}s")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES = [
    {
        "id": "single_country_mortality",
        "question": "What is the under-five mortality rate in Cambodia?",
        "expect_countries": ["Cambodia"],
        "expect_keywords": ["mortality", "Cambodia"],
    },
    {
        "id": "single_country_immunization",
        "question": "What is the DTP3 immunization rate in Thailand?",
        "expect_countries": ["Thailand"],
        "expect_keywords": ["immunization", "Thailand"],
    },
    {
        "id": "multi_country_comparison",
        "question": "Compare DTP3 immunization rates in Thailand, Cambodia and Laos",
        "expect_countries": ["Thailand", "Cambodia", "Laos"],
        "expect_n_dfs": 3,
    },
    {
        "id": "vague_query",
        "question": "Tell me about child health in Vietnam",
        "expect_countries": ["Vietnam"],
        "expect_keywords": ["Vietnam"],
    },
    {
        "id": "unknown_country",
        "question": "What is the under-five mortality rate in Wakanda?",
        "expect_countries": [],
        "expect_keywords": ["not found", "country"],
    },
    {
        "id": "non_health_query",
        "question": "What is the GDP growth rate in Cambodia?",
        "expect_countries": ["Cambodia"],
        "expect_keywords": [],  # May or may not find data — just shouldn't crash
    },
    {
        "id": "trend_question",
        "question": "How has measles immunization changed in Laos over the past 10 years?",
        "expect_countries": ["Laos"],
        "expect_keywords": ["measles", "Laos"],
    },
    {
        "id": "multi_country_mortality",
        "question": "Compare child mortality rates in Cambodia, Vietnam and Myanmar",
        "expect_countries": ["Cambodia", "Vietnam", "Myanmar"],
        "expect_n_dfs": 3,
    },
]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_agent_case(case):
    print(f"\n{'='*60}\n{case['id']}: {case['question']}\n{'='*60}")

    result = run_agent(case["question"])

    # Always check: agent produced an answer
    assert result["answer"], "Agent returned empty answer"
    print(f"Answer ({len(result['answer'])} chars): {result['answer'][:300]}...")

    # Tool call sequence
    tools_used = [t["tool"] for t in result["tool_calls"]]
    print(f"Tools called: {tools_used}")

    # Mandatory tools
    assert "search_data_dictionary" in tools_used, "search_data_dictionary was never called"
    assert "ask_user_to_select_indicator" in tools_used, "ask_user_to_select_indicator was never called"

    # Visualization
    if result["viz_spec"]:
        print(f"Viz spec: {result['viz_spec']}")
    else:
        print("WARNING: No viz spec generated")

    # Expected number of fetched DataFrames
    if "expect_n_dfs" in case:
        n = len(result["fetched_dfs"])
        print(f"Fetched DataFrames: {n} (expected {case['expect_n_dfs']})")
        assert n == case["expect_n_dfs"], f"Expected {case['expect_n_dfs']} DataFrames, got {n}"

    # Answer should mention expected keywords (case-insensitive)
    for kw in case.get("expect_keywords", []):
        assert kw.lower() in result["answer"].lower(), \
            f"Expected keyword '{kw}' not found in answer"

    print("PASS")
