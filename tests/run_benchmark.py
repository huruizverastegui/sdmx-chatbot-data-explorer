"""
UNICEF SDMX Chatbot — Benchmark runner.

Runs every flow in benchmark_scenarios.yaml against the live agent using the
same multi-turn loop as app.py (chat history injected between turns, indicator
selection auto-answered as "1").

Usage:
    conda run -n genai_vector python tests/run_benchmark.py
    conda run -n genai_vector python tests/run_benchmark.py --flows health_001 mortality_001
    conda run -n genai_vector python tests/run_benchmark.py --dry-run
    conda run -n genai_vector python tests/run_benchmark.py --report results.json
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import yaml
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import agents.sdmx_agent as agent_module
from agents.sdmx_agent import (
    AgentSession,
    SYSTEM_PROMPT,
    TOOLS,
    _TOOL_MAP,
    set_current_session,
    mark_new_question,
)

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    question: str
    expected_mode: str
    assertions: Dict[str, Any]
    tool_calls: List[Dict] = field(default_factory=list)
    final_answer: str = ""
    error: Optional[str] = None
    scores: Dict[str, Optional[bool]] = field(default_factory=dict)
    viz_checks: Dict[str, Any] = field(default_factory=dict)   # structural + LLM viz eval
    duration_s: float = 0.0


@dataclass
class FlowRecord:
    flow_id: str
    name: str
    program_area: str
    turns: List[TurnRecord] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        all_scores = [v for t in self.turns for v in t.scores.values() if v is not None]
        return sum(all_scores) / len(all_scores) if all_scores else 0.0

    @property
    def data_retrieved_rate(self) -> float:
        vals = [t.scores.get("data_retrieved") for t in self.turns if "data_retrieved" in t.scores]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0


# ── Azure OpenAI client ───────────────────────────────────────────────────────

def _get_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_LLM_API_VERSION"],
    )


# ── Multi-turn agent runner (mirrors app.py run_agent_until_pause) ────────────

def _run_agent_loop(
    agent_messages: list,
    turn_record: TurnRecord,
    patched_tool_map: dict,
) -> str:
    """
    Run the Azure OpenAI tool-use loop until the agent finishes or asks for
    indicator selection. Indicator selection is auto-answered "1".
    Records every tool call into turn_record.
    Returns the agent's final text answer.
    """
    client = _get_client()
    deployment = os.environ["AZURE_OPENAI_LLM_DEPLOYMENT"]

    while True:
        response = client.chat.completions.create(
            model=deployment,
            messages=agent_messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        agent_messages.append(msg)

        if not msg.tool_calls:
            return msg.content or ""

        pause_tc_id = None
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                agent_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Error parsing arguments: {e}",
                })
                continue

            if fn_name == "ask_user_to_select_indicator":
                # Record the call, auto-select option 1
                turn_record.tool_calls.append({
                    "name": fn_name,
                    "args": {"question": fn_args.get("question", "")[:200]},
                    "result": "User selected: 1 (auto-selected by benchmark)",
                })
                agent_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "User selected: 1",
                })
            else:
                fn = patched_tool_map.get(fn_name, _TOOL_MAP.get(fn_name))
                if fn is None:
                    result = f"Unknown tool: {fn_name}"
                else:
                    result = fn(**fn_args)
                snippet = str(result)[:300]
                turn_record.tool_calls.append({"name": fn_name, "args": fn_args, "result": snippet})
                agent_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })


def _extract_session_context(agent_messages: list) -> dict:
    """Mirror of app.py._extract_session_context."""
    countries, indicator_ids, indicator_names = [], [], []
    for msg in agent_messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                continue
            if tc.function.name == "query_sdmx_api":
                geo = args.get("geography_name")
                if geo and geo not in countries:
                    countries.append(geo)
                ind = args.get("indicator_id")
                if ind and ind not in indicator_ids:
                    indicator_ids.append(ind)
            elif tc.function.name == "run_quality_checks":
                iname = args.get("indicator_name")
                if iname and iname not in indicator_names:
                    indicator_names.append(iname)
    return {"countries": countries, "indicator_ids": indicator_ids, "indicator_names": indicator_names}


def _build_context_suffix(ctx: dict) -> str:
    if not ctx or not any(ctx.values()):
        return ""
    lines = ["[Prior session context — use this to answer follow-up questions]"]
    if ctx.get("countries"):
        lines.append(f"Countries discussed: {', '.join(ctx['countries'])}")
    if ctx.get("indicator_names"):
        lines.append(f"Indicator: {', '.join(ctx['indicator_names'])}")
    elif ctx.get("indicator_ids"):
        lines.append(f"Indicator IDs: {', '.join(ctx['indicator_ids'])}")
    return "\n\n" + "\n".join(lines)


def _recent_turns(chat_history: list, n_pairs: int = 2, max_chars: int = 500) -> list:
    turns = [m for m in chat_history if m["role"] in ("user", "assistant")]
    recent = turns[-(n_pairs * 2):]
    result = []
    for msg in recent:
        content = msg["content"]
        if msg["role"] == "assistant" and len(content) > max_chars:
            content = content[:max_chars] + "…"
        result.append({"role": msg["role"], "content": content})
    return result


def _build_agent_messages(prompt: str, chat_history: list, session_ctx: dict) -> list:
    context_suffix = _build_context_suffix(session_ctx)
    return [
        {"role": "system", "content": SYSTEM_PROMPT + context_suffix},
        *_recent_turns(chat_history),
        {"role": "user", "content": prompt},
    ]


# ── Visualization quality checks ──────────────────────────────────────────────

_STOPWORDS = {
    "what", "is", "the", "in", "for", "of", "a", "an", "and", "to", "me",
    "show", "can", "you", "i", "how", "does", "do", "it", "with", "by",
    "that", "this", "at", "on", "are", "was", "be", "have", "has",
}


def _structural_viz_checks(viz_spec: dict, dfs: list) -> dict:
    """Deterministic checks against the viz spec and fetched data."""
    import pandas as pd

    if not viz_spec:
        return {"viz_spec_exists": False}

    checks: dict = {"viz_spec_exists": True}
    if not dfs:
        checks["data_available"] = False
        return checks

    checks["data_available"] = True
    df = pd.concat(dfs, ignore_index=True)

    x_col     = viz_spec.get("x_column")
    y_col     = viz_spec.get("y_column")
    color_col = viz_spec.get("color_by")
    chart_type = viz_spec.get("chart_type", "")
    filters   = viz_spec.get("filters") or {}

    checks["x_col_exists"]     = bool(x_col and x_col in df.columns)
    checks["y_col_exists"]     = bool(y_col and y_col in df.columns)
    if color_col:
        checks["color_col_exists"] = color_col in df.columns

    # Apply filters
    filtered = df.copy()
    _RANGE_RE_CHECK = __import__("re").compile(r"^(>=|<=|>|<)\s*\d")
    bad_filters = []
    for col, val in filters.items():
        if col not in filtered.columns:
            bad_filters.append(col)
            continue
        val_str = str(val)
        # List and range filters are valid — only flag missing column or unmatched exact values
        if isinstance(val, list) or _RANGE_RE_CHECK.match(val_str):
            continue
        mask = filtered[col].astype(str) == val_str
        if mask.any():
            filtered = filtered[mask]
        else:
            bad_filters.append(f"{col}={val}")
    # Only warn about filters that reference missing columns — range/list syntax is valid
    if bad_filters:
        checks["filter_warnings"] = bad_filters

    checks["rows_after_filters"] = len(filtered)
    checks["data_nonempty"]      = len(filtered) > 0

    # y has numeric values
    if y_col and y_col in filtered.columns:
        obs = pd.to_numeric(filtered[y_col], errors="coerce")
        checks["y_has_values"] = int(obs.notna().sum())
    else:
        checks["y_has_values"] = 0

    # Color cardinality
    if color_col and color_col in filtered.columns:
        n = filtered[color_col].nunique()
        checks["color_n_unique"]       = n
        checks["color_cardinality_ok"] = n <= 8

    # Line chart should have multiple time points
    if chart_type == "line" and x_col and x_col in filtered.columns:
        checks["line_n_points"] = int(filtered[x_col].nunique())
        checks["line_has_trend"] = checks["line_n_points"] >= 2

    # Detect single time period in data — line chart here would be a single dot
    time_col = next(
        (c for c in ("TIME_PERIOD", "Year", "YEAR") if c in filtered.columns), None
    )
    if time_col:
        n_time = int(filtered[time_col].nunique())
        checks["n_time_periods"] = n_time
        if n_time == 1 and chart_type == "line":
            checks["single_year_line_warning"] = True

    return checks


def _llm_viz_eval(viz_spec: dict, dfs: list, question: str) -> dict:
    """
    Ask GPT to evaluate whether the chart spec sensibly answers the user's question.
    Returns a dict with boolean flags and an 'issues' string.
    """
    import pandas as pd

    if not viz_spec or not dfs:
        return {"skipped": "no viz spec or data"}

    df = pd.concat(dfs, ignore_index=True)
    sample_rows = df.head(3).to_string(index=False)
    col_list    = ", ".join(df.columns.tolist())

    prompt = f"""You are evaluating a data visualization produced by an AI chatbot.

User question: "{question}"

Visualization spec:
- Chart type : {viz_spec.get("chart_type")}
- X column   : {viz_spec.get("x_column")}
- Y column   : {viz_spec.get("y_column")}
- Color by   : {viz_spec.get("color_by")}
- Title      : {viz_spec.get("title")}
- Filters    : {json.dumps(viz_spec.get("filters") or {})}

Available columns in the data: {col_list}

Sample data (first 3 rows):
{sample_rows}

Evaluate on four dimensions. Respond with a JSON object only — no other text:
{{
  "chart_type_appropriate": true/false,   // line for trends, bar for comparisons/snapshots
  "title_matches_question": true/false,   // title describes what the chart shows re the question
  "axes_make_sense":        true/false,   // x/y columns are reasonable choices for this data
  "answers_question":       true/false,   // overall: would this chart actually answer the user's question
  "issues": "brief note on any problems, or empty string if none"
}}"""

    client = _get_client()
    deployment = os.environ["AZURE_OPENAI_LLM_DEPLOYMENT"]
    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


def evaluate_visualization(turn: TurnRecord, session: AgentSession) -> None:
    """Run structural + LLM checks and store results in turn.viz_checks."""
    viz_spec = session.viz_spec
    dfs      = session.fetched_dfs

    structural = _structural_viz_checks(viz_spec, dfs)
    turn.viz_checks.update(structural)

    # Only run LLM eval if the viz spec exists and data is available
    if structural.get("viz_spec_exists") and structural.get("data_available"):
        llm_eval = _llm_viz_eval(viz_spec, dfs, turn.question)
        turn.viz_checks["llm_eval"] = llm_eval


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_turn(turn: TurnRecord) -> None:
    calls = turn.tool_calls
    called = {c["name"] for c in calls}
    answer = turn.final_answer.lower()
    assertions = turn.assertions

    mode_detected = "A" if "search_data_dictionary" in called else "B"
    turn.scores["mode_correct"] = (mode_detected == turn.expected_mode)

    if assertions.get("data_retrieved"):
        api_calls = [c for c in calls if c["name"] == "query_sdmx_api"]
        got_data = any(
            "No data found" not in c["result"] and "404" not in c["result"]
            for c in api_calls
        )
        turn.scores["data_retrieved"] = bool(api_calls) and got_data

    if assertions.get("visualization_created"):
        turn.scores["visualization_created"] = "create_visualization" in called

    hints = assertions.get("indicator_hint", [])
    if hints:
        search_results = " ".join(
            c["result"].lower() for c in calls if c["name"] == "search_data_dictionary"
        )
        combined = search_results + answer
        turn.scores["indicator_hint"] = any(h.lower() in combined for h in hints)

    if assertions.get("multi_country"):
        geo_ids = {c["args"].get("geography_id", "") for c in calls if c["name"] == "query_sdmx_api"}
        turn.scores["multi_country"] = len(geo_ids) >= 2

    breakdown_kw = assertions.get("breakdown_requested", "")
    if breakdown_kw:
        viz_calls = [c for c in calls if c["name"] == "create_visualization"]
        breakdown_text = " ".join(json.dumps(c["args"]).lower() for c in viz_calls)
        turn.scores["breakdown_requested"] = breakdown_kw.lower() in breakdown_text

    # searches_before_clarifying: search_data_dictionary must be the very first tool call.
    # Fails if agent gives text-only response (no tools called) OR calls something else first.
    if assertions.get("searches_before_clarifying"):
        if calls:
            turn.scores["searches_before_clarifying"] = calls[0]["name"] == "search_data_dictionary"
        else:
            turn.scores["searches_before_clarifying"] = False

    # chart_type: create_visualization must be called with the specified chart_type value.
    expected_chart_type = assertions.get("chart_type")
    if expected_chart_type:
        viz_calls = [c for c in calls if c["name"] == "create_visualization"]
        if viz_calls:
            actual_type = (viz_calls[-1]["args"].get("chart_type") or "").lower()
            turn.scores["chart_type"] = actual_type == expected_chart_type.lower()
        else:
            turn.scores["chart_type"] = False

    turn.scores["no_error"] = turn.error is None


# ── Flow runner ───────────────────────────────────────────────────────────────

def run_flow(flow_def: dict) -> FlowRecord:
    record = FlowRecord(
        flow_id=flow_def["id"],
        name=flow_def["name"],
        program_area=flow_def["program_area"],
    )

    session = AgentSession()
    set_current_session(session)

    chat_history: list = []   # {role, content} pairs for context injection
    session_ctx: dict = {}    # countries/indicators extracted from prior turns

    for turn_def in flow_def["turns"]:
        turn = TurnRecord(
            question=turn_def["question"],
            expected_mode=turn_def.get("expected_mode", "A"),
            assertions=turn_def.get("assertions", {}),
        )
        record.turns.append(turn)

        agent_messages = _build_agent_messages(turn.question, chat_history, session_ctx)
        mark_new_question()

        t0 = time.time()
        try:
            answer = _run_agent_loop(agent_messages, turn, _TOOL_MAP)
            turn.final_answer = answer
        except Exception as exc:
            turn.error = str(exc)
            turn.final_answer = ""
        turn.duration_s = round(time.time() - t0, 1)

        # Viz quality checks (structural + LLM)
        if any(c["name"] == "create_visualization" for c in turn.tool_calls):
            evaluate_visualization(turn, session)

        # Update chat history and session context for next turn
        chat_history.append({"role": "user", "content": turn.question})
        chat_history.append({"role": "assistant", "content": turn.final_answer})
        session_ctx = _extract_session_context(agent_messages)

        _score_turn(turn)

    return record


# ── Reporting ─────────────────────────────────────────────────────────────────

_PASS = "✓"
_FAIL = "✗"
_SKIP = "–"


def _check(val: Optional[bool]) -> str:
    if val is True:  return _PASS
    if val is False: return _FAIL
    return _SKIP


def print_report(records: List[FlowRecord]) -> None:
    total_assertions = sum(
        len([v for v in t.scores.values() if v is not None])
        for r in records for t in r.turns
    )
    total_passed = sum(
        len([v for v in t.scores.values() if v is True])
        for r in records for t in r.turns
    )
    total_errors = sum(1 for r in records for t in r.turns if t.error)

    print("\n" + "═" * 70)
    print("SDMX CHATBOT BENCHMARK RESULTS")
    print("═" * 70)
    if total_assertions:
        print(f"Flows: {len(records)}  |  Total assertions: {total_assertions}"
              f"  |  Passed: {total_passed}  |  Overall: {total_passed / total_assertions:.0%}")
    if total_errors:
        print(f"Errors (turns that crashed): {total_errors}")
    print()

    by_area: Dict[str, List[FlowRecord]] = {}
    for r in records:
        by_area.setdefault(r.program_area, []).append(r)

    for area, flows in sorted(by_area.items()):
        area_scores = [v for f in flows for t in f.turns for v in t.scores.values() if v is not None]
        area_pct = f"{sum(area_scores)/len(area_scores):.0%}" if area_scores else "—"
        print(f"▸ {area} ({area_pct})")
        for flow in flows:
            status = _PASS if flow.pass_rate >= 0.75 else (_SKIP if flow.pass_rate >= 0.5 else _FAIL)
            print(f"  {status} [{flow.flow_id}] {flow.name} — {flow.pass_rate:.0%}")
            for i, turn in enumerate(flow.turns, 1):
                scores_str = "  ".join(f"{k}:{_check(v)}" for k, v in turn.scores.items())
                mode_actual = "A" if any(c["name"] == "search_data_dictionary" for c in turn.tool_calls) else "B"
                mode_ok = _check(turn.scores.get("mode_correct"))
                err_str = f" ⚠ {turn.error[:60]}" if turn.error else ""
                print(f"    Turn {i} [mode {mode_actual}{mode_ok}] {turn.duration_s}s  {scores_str}{err_str}")
                if turn.viz_checks:
                    vc = turn.viz_checks
                    struct = (
                        f"rows={vc.get('rows_after_filters','?')}  "
                        f"y_values={vc.get('y_has_values','?')}  "
                        + (f"color_unique={vc.get('color_n_unique','?')}  " if 'color_n_unique' in vc else "")
                        + (f"line_pts={vc.get('line_n_points','?')}  " if 'line_n_points' in vc else "")
                        + (f"time_periods={vc.get('n_time_periods','?')}  " if 'n_time_periods' in vc else "")
                        + ("⚠ SINGLE_YEAR_LINE  " if vc.get('single_year_line_warning') else "")
                        + (f"filters⚠={vc['filter_warnings']}" if vc.get('filter_warnings') else "")
                    )
                    llm = vc.get("llm_eval", {})
                    if llm and "error" not in llm and "skipped" not in llm:
                        lflags = "  ".join(
                            f"{k[:12]}:{_check(v)}"
                            for k, v in llm.items()
                            if k != "issues" and isinstance(v, bool)
                        )
                        issues = f"  → {llm['issues']}" if llm.get("issues") else ""
                        print(f"      viz  {struct}")
                        print(f"      llm  {lflags}{issues}")
                    elif llm.get("error"):
                        print(f"      viz  {struct}  llm_error={llm['error'][:60]}")
                    else:
                        print(f"      viz  {struct}")
        print()

    print("Data retrieval success by program area:")
    print(f"  {'Area':<25} {'Flows':>6}  {'Retrieved':>10}")
    for area, flows in sorted(by_area.items()):
        dr_rates = [f.data_retrieved_rate for f in flows
                    if any(t.assertions.get("data_retrieved") for t in f.turns)]
        avg = f"{sum(dr_rates)/len(dr_rates):.0%}" if dr_rates else "—"
        print(f"  {area:<25} {len(flows):>6}  {avg:>10}")
    print()


def save_json(records: List[FlowRecord], path: Path) -> None:
    def _ser(obj):
        if isinstance(obj, FlowRecord):
            return {
                "flow_id": obj.flow_id, "name": obj.name,
                "program_area": obj.program_area,
                "pass_rate": round(obj.pass_rate, 3),
                "turns": [_ser(t) for t in obj.turns],
            }
        if isinstance(obj, TurnRecord):
            return {
                "question": obj.question,
                "expected_mode": obj.expected_mode,
                "duration_s": obj.duration_s,
                "error": obj.error,
                "scores": obj.scores,
                "viz_checks": obj.viz_checks,
                "tool_calls": [{"name": c["name"], "args": c["args"]} for c in obj.tool_calls],
                "final_answer": obj.final_answer[:500],
            }
        return str(obj)

    with open(path, "w") as f:
        json.dump([_ser(r) for r in records], f, indent=2)
    print(f"Results saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run SDMX chatbot benchmark")
    parser.add_argument("--flows", nargs="*", help="Flow IDs to run (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Validate YAML only")
    parser.add_argument("--report", metavar="PATH", help="Save JSON results to file")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between flows (rate limiting)")
    args = parser.parse_args()

    scenarios_path = ROOT / "tests" / "benchmark_scenarios.yaml"
    with open(scenarios_path) as f:
        data = yaml.safe_load(f)

    all_flows = data["flows"]
    if args.flows:
        all_flows = [fl for fl in all_flows if fl["id"] in args.flows]
        if not all_flows:
            sys.exit(f"No flows matched: {args.flows}")

    print(f"Benchmark: {len(all_flows)} flows loaded from {scenarios_path.name}")

    if args.dry_run:
        for fl in all_flows:
            print(f"  [{fl['id']}] {fl['name']} — {len(fl.get('turns', []))} turn(s)")
        print("Dry run complete — YAML is valid.")
        return

    records: List[FlowRecord] = []
    for i, fl in enumerate(all_flows, 1):
        print(f"\n[{i}/{len(all_flows)}] Running: {fl['id']} — {fl['name']}")
        record = run_flow(fl)
        records.append(record)
        if i < len(all_flows):
            time.sleep(args.delay)

    print_report(records)

    if args.report:
        save_json(records, Path(args.report))


if __name__ == "__main__":
    main()
