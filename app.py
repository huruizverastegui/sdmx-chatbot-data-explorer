"""
Streamlit UI for the UNICEF SDMX agent.

Run with:
  conda run -n genai_vector streamlit run app.py
"""

import json
import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

from agents.sdmx_agent import (
    SYSTEM_PROMPT, TOOLS, _TOOL_MAP,
    AgentSession, set_current_session,
    get_fetched_dfs, get_source_urls, get_viz_spec, reset_session_data, mark_new_question,
)

load_dotenv()

st.set_page_config(page_title="UNICEF SDMX Explorer", page_icon="🌍", layout="centered")

# ---------------------------------------------------------------------------
# Startup config validation
# ---------------------------------------------------------------------------

_REQUIRED_ENV = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_LLM_DEPLOYMENT",
    "AZURE_OPENAI_LLM_API_VERSION",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_API_VERSION",
]
_missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
if _missing:
    st.error(
        f"Missing required environment variables: {', '.join(_missing)}\n\n"
        "Set them in Azure App Service → Settings → Environment variables."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Password gate
# ---------------------------------------------------------------------------

def _check_password() -> bool:
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected:
        return True  # no password configured — open access

    if st.session_state.get("authenticated"):
        return True

    st.title("🌍 UNICEF SDMX Data Explorer")
    pwd = st.text_input("Password", type="password", key="pwd_input")
    if st.button("Sign in"):
        if pwd == expected:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not _check_password():
    st.stop()

st.title("🌍 UNICEF SDMX Data Explorer")


# ---------------------------------------------------------------------------
# Agent loop (Streamlit-aware — pauses at ask_user_to_select_indicator)
# ---------------------------------------------------------------------------

def _get_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_LLM_API_VERSION"],
    )


def run_agent_until_pause(agent_messages: list) -> dict:
    """
    Run the agent loop until it either finishes or requests user input.

    Returns:
      {"status": "done",               "answer": str}
      {"status": "awaiting_selection", "question": str, "tool_call_id": str}
    """
    client = _get_client()
    deployment = os.environ["AZURE_OPENAI_LLM_DEPLOYMENT"]

    while True:
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=agent_messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "not set")
            api_version = os.environ.get("AZURE_OPENAI_LLM_API_VERSION", "not set")
            st.error(
                f"Azure OpenAI error: {e}\n\n"
                f"**Endpoint:** `{endpoint}`  \n"
                f"**Deployment:** `{deployment}`  \n"
                f"**API version:** `{api_version}`"
            )
            st.stop()
        msg = response.choices[0].message
        agent_messages.append(msg)

        if not msg.tool_calls:
            return {"status": "done", "answer": msg.content}

        pause = None
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                agent_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Error: could not parse tool arguments — {e}. Please retry with valid JSON.",
                })
                continue

            if fn_name == "ask_user_to_select_indicator":
                pause = {"tool_call_id": tc.id, "question": fn_args["question"]}
            else:
                result = _TOOL_MAP[fn_name](**fn_args)
                agent_messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

        if pause:
            return {"status": "awaiting_selection", **pause}


# ---------------------------------------------------------------------------
# Data visualisation
# ---------------------------------------------------------------------------

_TOTAL_VALUES = {"_T", "Total", "TOTAL", "_ALL", "All"}

# Columns that are not disaggregation dimensions — never filter these
_NON_DISAGG_COLS = {
    "TIME_PERIOD", "Year", "YEAR",
    "OBS_VALUE", "LOWER_BOUND", "UPPER_BOUND", "OBS_STATUS", "Observation Status",
    "REF_AREA", "Geographic area", "Reference Areas",
    "INDICATOR", "Indicator",
    "UNIT_MEASURE", "Unit of measure",
    "DATA_SOURCE", "SHORT_SOURCE", "CDDATASOURCE", "PROVIDER",
    "DEFINITION", "MDG_REGION", "REF_PERIOD", "COVERAGE_TIME",
    "OBS_VALUE_CHARACTER", "OBS_CONF", "OBS_FOOTNOTE",
    "COUNTRY_NOTES", "SERIES_FOOTNOTE", "SOURCE_LINK",
}


def _resolve_col(df: pd.DataFrame, requested: "str | None", fallbacks: list) -> "str | None":
    """Return `requested` if it exists in df, else the first fallback that does, else None."""
    if requested and requested in df.columns:
        return requested
    for col in fallbacks:
        if col in df.columns:
            return col
    return None


_GEO_COLS = {"Geographic area", "Reference Areas", "REF_AREA", "GEOGRAPHIC_AREA"}


def _resolve_duplicates(
    df: pd.DataFrame, x: str, y: str, color: "str | None"
) -> "tuple[pd.DataFrame, list[str]]":
    """
    If multiple rows exist per (x, color_by, geography) group, find disaggregation columns
    that have a 'Total' / '_T' value and filter to them. Geography columns are always included
    in the groupby so multi-country data is never mistaken for duplicates.
    """
    geo_cols = [c for c in _GEO_COLS if c in df.columns]
    group_cols = list(dict.fromkeys(             # ordered, deduplicated
        [c for c in [x, color] if c and c in df.columns] + geo_cols
    ))
    applied: list = []

    if not group_cols or df.groupby(group_cols)[y].count().max() <= 1:
        return df, applied

    result = df.copy()
    candidate_cols = [
        c for c in df.columns
        if c not in _NON_DISAGG_COLS and c not in set(group_cols) and c != y
    ]

    for col in candidate_cols:
        if result.groupby(group_cols)[y].count().max() <= 1:
            break
        unique_vals = set(result[col].dropna().unique())
        total_val = next((v for v in unique_vals if str(v) in _TOTAL_VALUES), None)
        if total_val is not None and len(unique_vals) > 1:
            result = result[result[col] == total_val]
            applied.append(f"{col}='{total_val}'")

    return result, applied


def _render_chart(dfs: list, spec: dict) -> None:
    if not dfs:
        return

    combined = pd.concat(dfs, ignore_index=True)

    x = spec["x_column"]
    y = spec["y_column"]
    color = spec.get("color_by")

    # Apply filters specified by the LLM, but never filter on the color_by column
    # (that would collapse the breakdown the user asked for)
    color_lower = color.lower() if color else ""
    for col, val in spec.get("filters", {}).items():
        if col.lower() == color_lower:
            continue  # skip — this column is the breakdown dimension
        if col in combined.columns:
            filtered = combined[combined[col] == val]
            if not filtered.empty:
                combined = filtered
    title = spec.get("title", "")
    chart_type = spec.get("chart_type", "line")

    # Resolve column names — the LLM may use aliases or invent names
    x = _resolve_col(combined, x, ["TIME_PERIOD", "Year", "YEAR", "Geographic area", "Reference Areas", "REF_AREA"])
    y = _resolve_col(combined, y, ["OBS_VALUE"])
    if color:
        color = _resolve_col(combined, color, ["Geographic area", "Reference Areas", "REF_AREA", "Sex", "SEX", "Residence"])

    if not x or not y:
        st.warning(f"Could not resolve chart columns. Available: {list(combined.columns)}")
        return

    combined[y] = pd.to_numeric(combined[y], errors="coerce")
    combined = combined.dropna(subset=[y])

    # Resolve any remaining duplicates per (x, color_by) by filtering to Total rows
    combined, auto_filters = _resolve_duplicates(combined, x, y, color)
    if auto_filters:
        st.caption(f"Showing aggregate totals — auto-filtered: {', '.join(auto_filters)}")


    # Safeguard: cap high-cardinality color dimensions to top 8 by mean value
    MAX_COLOR_SERIES = 8
    if color and color in combined.columns:
        unique_vals = combined[color].dropna().unique()
        if len(unique_vals) > MAX_COLOR_SERIES:
            top_vals = (
                combined.groupby(color)[y].mean()
                .nlargest(MAX_COLOR_SERIES)
                .index.tolist()
            )
            combined = combined[combined[color].isin(top_vals)]
            st.caption(
                f"Too many series to display clearly — showing top {MAX_COLOR_SERIES} "
                f"by average {y} (out of {len(unique_vals)} values in '{color}')."
            )

    unit_col = next((c for c in ("Unit of measure", "UNIT_MEASURE") if c in combined.columns), None)
    y_label = combined[unit_col].iloc[0] if unit_col and unit_col in combined.columns else y

    if chart_type == "bar":
        fig = px.bar(
            combined, x=x, y=y, color=color,
            barmode="group", title=title,
            labels={y: y_label, x: x},
            template="plotly_white",
        )
    else:
        fig = px.line(
            combined, x=x, y=y, color=color,
            markers=True, title=title,
            labels={y: y_label, x: x},
            template="plotly_white",
        )

    fig.update_layout(hovermode="x unified", legend_title=color or "")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Conversation memory (Option A + C)
# ---------------------------------------------------------------------------

def _extract_session_context(agent_messages: list) -> dict:
    """
    Scan completed agent_messages for structured facts extracted from tool calls.
    Works on the raw list which contains both dicts (tool results) and
    ChatCompletionMessage objects (model responses).
    """
    countries, indicator_ids, indicator_names = [], [], []

    for msg in agent_messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                continue

            if name == "query_sdmx_api":
                geo = args.get("geography_name")
                if geo and geo not in countries:
                    countries.append(geo)
                ind = args.get("indicator_id")
                if ind and ind not in indicator_ids:
                    indicator_ids.append(ind)

            elif name == "run_quality_checks":
                iname = args.get("indicator_name")
                if iname and iname not in indicator_names:
                    indicator_names.append(iname)

    return {
        "countries": countries,
        "indicator_ids": indicator_ids,
        "indicator_names": indicator_names,
    }


def _build_context_suffix(ctx: dict) -> str:
    """Return a compact context block to append to the system prompt."""
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
    """
    Return the last n_pairs of user/assistant exchanges as plain dicts,
    trimming long assistant answers so tool payloads don't bloat the context.
    """
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


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "chat_history": [],     # entries: {role, content, chart?}
        "agent_messages": None,
        "awaiting_selection": False,
        "selection_question": "",
        "pending_tool_call_id": None,
        "session_ctx": {},
        "agent_session": AgentSession(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()
set_current_session(st.session_state.agent_session)


# ---------------------------------------------------------------------------
# Parse numbered options out of the model's selection question
# ---------------------------------------------------------------------------

def _parse_options(question: str) -> list:
    lines = re.findall(r"^\s*\d+\.\s+.+", question, re.MULTILINE)
    return [re.sub(r"^\s*\d+\.\s+", "", ln).strip() for ln in lines]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attach_chart_to_last_message() -> None:
    """Store current chart data inside the last assistant chat_history entry."""
    spec = get_viz_spec()
    dfs = get_fetched_dfs()
    if not spec or not dfs:
        return
    chart_data = {"spec": spec, "dfs": dfs, "urls": get_source_urls()}
    for msg in reversed(st.session_state.chat_history):
        if msg["role"] == "assistant":
            msg["chart"] = chart_data
            return


def _render_chart_block(chart: dict, key: str) -> None:
    """Render chart + download button + source expander from a stored chart dict."""
    _render_chart(chart["dfs"], chart["spec"])
    combined_csv = pd.concat(chart["dfs"], ignore_index=True)
    st.download_button(
        label="Download data (CSV)",
        data=combined_csv.to_csv(index=False),
        file_name="unicef_data.csv",
        mime="text/csv",
        key=f"dl_{key}",
    )
    if chart.get("urls"):
        with st.expander("Raw SDMX data sources"):
            for entry in chart["urls"]:
                label = f"{entry['geography_id']} — {entry['indicator_id']}"
                source_line = f"  \n*Source: {entry['data_source']}*" if entry.get("data_source") else ""
                st.markdown(f"**{label}**{source_line}  \n{entry['url']}")


# ---------------------------------------------------------------------------
# Render chat history (messages + inline charts)
# ---------------------------------------------------------------------------

for _i, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        if "chart" in msg:
            urls = msg["chart"].get("urls", [])
            if urls:
                parts = []
                for entry in urls:
                    src = entry.get("data_source")
                    geo = entry.get("geography_id", "")
                    if src:
                        parts.append(f"{geo}: {src}" if len(urls) > 1 else src)
                    else:
                        parts.append(geo)
                unique_parts = list(dict.fromkeys(parts))  # preserve order, deduplicate
                st.markdown(f"**Source:** {' · '.join(unique_parts)}")
                st.divider()
        st.markdown(msg["content"])
    if "chart" in msg:
        _render_chart_block(msg["chart"], key=str(_i))


# ---------------------------------------------------------------------------
# Phase 2 — user selects an indicator
# ---------------------------------------------------------------------------

if st.session_state.awaiting_selection:
    options = _parse_options(st.session_state.selection_question)

    with st.chat_message("assistant"):
        preamble = re.split(r"\n\s*1\.", st.session_state.selection_question)[0].strip()
        st.markdown(preamble)

        if options:
            chosen_text = st.radio(
                "Select an indicator:",
                options=options,
                index=0,
                key="indicator_radio",
            )
        else:
            chosen_text = st.text_input("Your choice:", key="indicator_text")

        confirmed = st.button("Confirm selection")

    if confirmed and chosen_text:
        st.session_state.chat_history.append(
            {"role": "user", "content": f"I'd like to see: **{chosen_text}**"}
        )
        st.session_state.agent_messages.append({
            "role": "tool",
            "tool_call_id": st.session_state.pending_tool_call_id,
            "content": f"User selected: {chosen_text}",
        })
        st.session_state.awaiting_selection = False

        with st.spinner("Fetching data and running quality checks…"):
            result = run_agent_until_pause(st.session_state.agent_messages)

        if result["status"] == "done":
            st.session_state.chat_history.append(
                {"role": "assistant", "content": result["answer"]}
            )
            _attach_chart_to_last_message()
            st.session_state.session_ctx = _extract_session_context(
                st.session_state.agent_messages
            )
            st.session_state.agent_messages = None
        elif result["status"] == "awaiting_selection":
            st.session_state.awaiting_selection = True
            st.session_state.selection_question = result["question"]
            st.session_state.pending_tool_call_id = result["tool_call_id"]

        st.rerun()


# ---------------------------------------------------------------------------
# Phase 1 — user asks a question
# ---------------------------------------------------------------------------

else:
    if prompt := st.chat_input("Ask about UNICEF data… e.g. 'What is the immunization rate in Thailand?'"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        mark_new_question()  # lazy reset — chart stays until new data actually arrives

        st.session_state.agent_messages = _build_agent_messages(
            prompt,
            st.session_state.chat_history[:-1],  # exclude the just-appended user msg
            st.session_state.session_ctx,
        )

        with st.spinner("Searching indicators…"):
            result = run_agent_until_pause(st.session_state.agent_messages)

        if result["status"] == "awaiting_selection":
            st.session_state.awaiting_selection = True
            st.session_state.selection_question = result["question"]
            st.session_state.pending_tool_call_id = result["tool_call_id"]
        elif result["status"] == "done":
            st.session_state.chat_history.append(
                {"role": "assistant", "content": result["answer"]}
            )
            _attach_chart_to_last_message()
            st.session_state.session_ctx = _extract_session_context(
                st.session_state.agent_messages
            )
            st.session_state.agent_messages = None

        st.rerun()
