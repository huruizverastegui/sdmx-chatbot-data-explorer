"""
Integration tests — hit the real Azure OpenAI and SDMX APIs.
Run after major changes to catch regressions in agent behaviour.

Run all:
  conda run -n genai_vector python -m pytest tests/test_integration.py -v

Run one case:
  conda run -n genai_vector python -m pytest tests/test_integration.py -v -k single_country
"""
import json
import os

import pytest
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

from agents.sdmx_agent import (
    SYSTEM_PROMPT, TOOLS, _TOOL_MAP,
    AgentSession, set_current_session,
    get_fetched_dfs, get_viz_spec,
)


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def _auto_select(question: str) -> str:
    return "User selected: 1"


def run_agent(messages: list, session: AgentSession, timeout_s: int = 120) -> dict:
    import time
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_LLM_API_VERSION"],
    )
    deployment = os.environ["AZURE_OPENAI_LLM_DEPLOYMENT"]
    tool_map = dict(_TOOL_MAP, ask_user_to_select_indicator=_auto_select)

    tool_calls_log = []
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        resp = client.chat.completions.create(
            model=deployment, messages=messages, tools=TOOLS, tool_choice="auto"
        )
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return {
                "answer": msg.content,
                "fetched_dfs": get_fetched_dfs(),
                "viz_spec": get_viz_spec(),
                "tools": tool_calls_log,
            }

        for tc in msg.tool_calls:
            fn = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool_calls_log.append(fn)
            result = tool_map[fn](**args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    raise TimeoutError(f"Agent did not finish within {timeout_s}s")


def fresh_messages(question: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


@pytest.fixture
def session():
    s = AgentSession()
    set_current_session(s)
    return s


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestSingleCountry:
    def test_returns_answer_and_data(self, session):
        result = run_agent(fresh_messages("What is the DTP3 immunization rate in Cambodia?"), session)
        assert result["answer"], "No answer returned"
        assert len(result["fetched_dfs"]) >= 1, "No data fetched"

    def test_correct_tool_sequence(self, session):
        result = run_agent(fresh_messages("Under-five mortality rate in Thailand"), session)
        tools = result["tools"]
        assert "search_data_dictionary" in tools
        assert "ask_user_to_select_indicator" in tools
        assert "query_sdmx_api" in tools
        assert "get_data_dimensions" in tools

    def test_viz_spec_generated(self, session):
        result = run_agent(fresh_messages("Child mortality trend in Cambodia"), session)
        assert result["viz_spec"] is not None, "No visualization spec produced"
        assert result["viz_spec"]["y_column"] == "OBS_VALUE"


class TestMultiCountry:
    def test_fetches_data_for_each_country(self, session):
        result = run_agent(
            fresh_messages("Compare DTP3 immunization in Thailand, Cambodia and Vietnam"), session
        )
        assert len(result["fetched_dfs"]) == 3, (
            f"Expected 3 DataFrames, got {len(result['fetched_dfs'])}"
        )

    def test_answer_mentions_all_countries(self, session):
        result = run_agent(
            fresh_messages("Child mortality in Cambodia and Thailand"), session
        )
        answer_lower = result["answer"].lower()
        assert "cambodia" in answer_lower
        assert "thailand" in answer_lower


class TestFollowUpBreakdown:
    """MODE B: follow-up dimension questions must not trigger a new SDMX fetch."""

    def test_no_new_fetch_on_breakdown_question(self, session):
        # Turn 1 — initial fetch
        messages = fresh_messages("Child mortality in Cambodia")
        turn1 = run_agent(messages, session)
        n_dfs_after_turn1 = len(get_fetched_dfs())

        # Turn 2 — breakdown follow-up
        messages.append({"role": "assistant", "content": turn1["answer"]})
        messages.append({"role": "user", "content": "Show me this broken down by sex"})
        turn2 = run_agent(messages, session)

        assert "query_sdmx_api" not in turn2["tools"], (
            "query_sdmx_api must not be called on a breakdown follow-up"
        )
        assert len(get_fetched_dfs()) == n_dfs_after_turn1, (
            "Number of fetched DataFrames should not change on follow-up"
        )

    def test_viz_spec_updated_with_color_by(self, session):
        messages = fresh_messages("Child mortality in Cambodia")
        turn1 = run_agent(messages, session)

        messages.append({"role": "assistant", "content": turn1["answer"]})
        messages.append({"role": "user", "content": "Show me this broken down by sex"})
        turn2 = run_agent(messages, session)

        spec = get_viz_spec()
        assert spec is not None
        assert spec.get("color_by") is not None, "color_by should be set for sex breakdown"


class TestEdgeCases:
    def test_unknown_country_handled_gracefully(self, session):
        result = run_agent(fresh_messages("Child mortality in Wakanda"), session)
        assert result["answer"], "Should return an answer even for unknown country"
        answer_lower = result["answer"].lower()
        assert any(w in answer_lower for w in ("not found", "unavailable", "couldn't", "cannot", "no data"))

    def test_no_data_does_not_crash(self, session):
        """Querying a non-UNICEF indicator should not raise an exception."""
        result = run_agent(fresh_messages("GDP growth rate in Cambodia"), session)
        assert result["answer"], "Agent should answer gracefully even without SDMX data"

    def test_indicator_mismatch_not_displayed(self, session):
        """Data for a different indicator should never be shown to the user."""
        result = run_agent(fresh_messages("Child marriage rate in Thailand"), session)
        for df in result["fetched_dfs"]:
            if "INDICATOR" in df.columns:
                indicators = df["INDICATOR"].dropna().astype(str).str.upper().unique()
                for ind in indicators:
                    assert "IM_" not in ind, (
                        f"Immunization indicator {ind} shown for child marriage query"
                    )


# ---------------------------------------------------------------------------
# Realistic user question scenarios
# ---------------------------------------------------------------------------

SINGLE_TURN_SCENARIOS = [
    {
        "id": "dtp3_single_country",
        "question": "What is the DTP3 immunization rate in Thailand?",
        "expect_countries": ["thailand"],
        "expect_keywords": ["immunization", "thailand"],
        "expect_data": True,
    },
    {
        "id": "u5mr_single_country",
        "question": "What is the under-five mortality rate in Cambodia?",
        "expect_countries": ["cambodia"],
        "expect_keywords": ["mortality", "cambodia"],
        "expect_data": True,
    },
    {
        "id": "multi_country_comparison",
        "question": "Compare DTP3 immunization rates in Thailand, Cambodia and Vietnam",
        "expect_countries": ["thailand", "cambodia", "vietnam"],
        "expect_keywords": ["immunization"],
        "expect_n_dfs": 3,
        "expect_data": True,
    },
    {
        "id": "trend_question",
        "question": "How has measles immunization changed in Laos over the past 10 years?",
        "expect_countries": ["laos"],
        "expect_keywords": ["measles", "laos"],
        "expect_data": True,
    },
    {
        "id": "vague_health_query",
        "question": "Tell me about child health in Vietnam",
        "expect_countries": ["vietnam"],
        "expect_keywords": ["vietnam"],
        "expect_data": True,
    },
    {
        "id": "unknown_country",
        "question": "What is the under-five mortality rate in Wakanda?",
        "expect_countries": [],
        "expect_keywords": [],
        "expect_data": False,
    },
    {
        "id": "multi_country_mortality",
        "question": "Compare child mortality rates in Cambodia, Vietnam and Myanmar",
        "expect_countries": ["cambodia", "vietnam", "myanmar"],
        "expect_keywords": ["mortality"],
        "expect_n_dfs": 3,
        "expect_data": True,
    },
    {
        "id": "stunting_query",
        "question": "What is the stunting rate among children in Indonesia?",
        "expect_countries": ["indonesia"],
        "expect_keywords": ["indonesia"],
        "expect_data": True,
    },
    {
        "id": "wash_query",
        "question": "What percentage of the population in Myanmar has access to clean water?",
        "expect_countries": ["myanmar"],
        "expect_keywords": ["myanmar"],
        "expect_data": True,
    },
    {
        "id": "non_health_query",
        "question": "What is the GDP growth rate in Cambodia?",
        "expect_countries": ["cambodia"],
        "expect_keywords": [],
        "expect_data": False,  # May or may not find data — just shouldn't crash
    },
]

TWO_TURN_SCENARIOS = [
    {
        "id": "sex_breakdown",
        "turn1": "What is child mortality in Cambodia?",
        "turn2": "Can you show me this broken down by sex?",
        "expect_no_refetch": True,
        "expect_color_by": True,
    },
    {
        "id": "country_then_compare",
        "turn1": "What is the DTP3 immunization rate in Thailand?",
        "turn2": "Can you also show me Cambodia for comparison?",
        "expect_no_refetch": False,   # new country → re-fetch expected
        "expect_color_by": False,
    },
    {
        "id": "trend_then_breakdown",
        "turn1": "Show me stunting rates in Vietnam over time",
        "turn2": "Break it down by residence (urban vs rural)",
        "expect_no_refetch": True,
        "expect_color_by": True,
    },
]


@pytest.mark.parametrize("case", SINGLE_TURN_SCENARIOS, ids=[c["id"] for c in SINGLE_TURN_SCENARIOS])
def test_user_scenario_single_turn(case, session):
    print(f"\n{'='*60}\n{case['id']}: {case['question']}\n{'='*60}")

    result = run_agent(fresh_messages(case["question"]), session)

    assert result["answer"], "Agent returned empty answer"
    print(f"Answer ({len(result['answer'])} chars): {result['answer'][:300]}")
    print(f"Tools: {result['tools']}")

    # Mandatory tool sequence
    assert "search_data_dictionary" in result["tools"]
    assert "ask_user_to_select_indicator" in result["tools"]

    # Countries mentioned in answer
    answer_lower = result["answer"].lower()
    for country in case.get("expect_countries", []):
        assert country in answer_lower, f"Expected '{country}' in answer"

    # Keywords
    for kw in case.get("expect_keywords", []):
        assert kw.lower() in answer_lower, f"Expected keyword '{kw}' in answer"

    # Data fetched
    if case.get("expect_data"):
        assert len(result["fetched_dfs"]) >= 1, "Expected data to be fetched"
        assert result["viz_spec"] is not None, "Expected viz spec when data is fetched"

    # Exact DataFrame count
    if "expect_n_dfs" in case:
        n = len(result["fetched_dfs"])
        assert n == case["expect_n_dfs"], f"Expected {case['expect_n_dfs']} DataFrames, got {n}"

    print("PASS")


@pytest.mark.parametrize("case", TWO_TURN_SCENARIOS, ids=[c["id"] for c in TWO_TURN_SCENARIOS])
def test_user_scenario_two_turn(case, session):
    print(f"\n{'='*60}\n{case['id']}\n{'='*60}")

    # Turn 1
    messages = fresh_messages(case["turn1"])
    turn1 = run_agent(messages, session)
    assert turn1["answer"], "Turn 1 returned empty answer"
    dfs_after_turn1 = len(get_fetched_dfs())
    print(f"Turn 1 tools: {turn1['tools']}, DFs: {dfs_after_turn1}")

    # Turn 2
    messages.append({"role": "assistant", "content": turn1["answer"]})
    messages.append({"role": "user", "content": case["turn2"]})
    turn2 = run_agent(messages, session)
    assert turn2["answer"], "Turn 2 returned empty answer"
    print(f"Turn 2 tools: {turn2['tools']}")

    if case["expect_no_refetch"]:
        assert "query_sdmx_api" not in turn2["tools"], (
            "query_sdmx_api should not be called on a breakdown follow-up"
        )
        assert len(get_fetched_dfs()) == dfs_after_turn1, (
            "Number of fetched DataFrames should not increase on follow-up"
        )

    if case["expect_color_by"]:
        spec = get_viz_spec()
        assert spec and spec.get("color_by"), "Expected color_by to be set in viz spec"

    print("PASS")
