"""
Unit tests — no API calls, run in seconds.

Run:
  conda run -n genai_vector python -m pytest tests/test_unit.py -v
"""
import numpy as np
import pandas as pd
import pytest

from agents.sdmx_agent import (
    AgentSession, set_current_session,
    _data_matches_query, get_data_dimensions,
    get_fetched_dfs, get_viz_spec, get_source_urls,
    mark_new_question, reset_session_data,
    _resolve_country, _format_csv_response,
)

# Inline imports from app layer (pure functions, no Streamlit)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import _resolve_duplicates, _NON_DISAGG_COLS


# ===========================================================================
# Session isolation
# ===========================================================================

class TestSessionIsolation:
    def test_two_sessions_do_not_share_fetched_dfs(self):
        s1, s2 = AgentSession(), AgentSession()

        set_current_session(s1)
        s1.fetched_dfs.append(pd.DataFrame({"v": [1]}))

        set_current_session(s2)
        assert get_fetched_dfs() == [], "s2 should see its own empty list"

        set_current_session(s1)
        assert len(get_fetched_dfs()) == 1, "s1 data should still be there"

    def test_mark_new_question_scoped_to_session(self):
        s1, s2 = AgentSession(), AgentSession()
        set_current_session(s1)
        mark_new_question()
        set_current_session(s2)
        assert not s2.pending_reset, "pending_reset must not bleed to s2"

    def test_reset_session_data_clears_only_current_session(self):
        s1, s2 = AgentSession(), AgentSession()
        s1.fetched_dfs.append(pd.DataFrame({"v": [99]}))
        s2.fetched_dfs.append(pd.DataFrame({"v": [42]}))

        set_current_session(s1)
        reset_session_data()

        assert s1.fetched_dfs == []
        assert len(s2.fetched_dfs) == 1, "s2 must be untouched"


# ===========================================================================
# Indicator validation
# ===========================================================================

class TestDataMatchesQuery:
    """_data_matches_query falls back to True when embeddings aren't loaded (unit test context)."""

    def _dummy_vec(self) -> np.ndarray:
        v = np.ones(3, dtype=np.float32)
        return v / np.linalg.norm(v)

    def test_no_indicator_column_trusted(self):
        df = pd.DataFrame({"OBS_VALUE": [1, 2, 3], "TIME_PERIOD": [2020, 2021, 2022]})
        assert _data_matches_query(df, self._dummy_vec())

    def test_unknown_indicator_code_trusted(self):
        # Code not in the embeddings index → can't judge → trust it
        df = pd.DataFrame({"INDICATOR": ["UNKNOWN_XYZ"] * 2, "OBS_VALUE": [5, 6]})
        assert _data_matches_query(df, self._dummy_vec())


# ===========================================================================
# Duplicate resolution
# ===========================================================================

class TestResolveDuplicates:
    def test_single_country_sex_breakdown_not_filtered(self, sample_df_single):
        """Multi-row per (TIME, Sex) only because of M/F/T — not real duplicates."""
        result, applied = _resolve_duplicates(sample_df_single, "TIME_PERIOD", "OBS_VALUE", "Sex")
        assert applied == [], "Should not filter when each (TIME, Sex, Geo) is unique"
        assert len(result) == len(sample_df_single)

    def test_multi_country_sex_breakdown_not_filtered(self, sample_df_multi):
        """3 countries × 3 sex values = 9 rows per year — geography makes them unique."""
        result, applied = _resolve_duplicates(sample_df_multi, "TIME_PERIOD", "OBS_VALUE", "Sex")
        assert applied == [], "Geography disambiguates — no filter should be applied"
        assert len(result) == len(sample_df_multi)

    def test_genuine_duplicates_are_filtered(self):
        """Same (TIME, Sex, Geo) repeated — one has Total wealth quintile, one has Q1."""
        df = pd.DataFrame({
            "TIME_PERIOD":     [2020, 2020],
            "OBS_VALUE":       [10.0, 8.0],
            "Geographic area": ["Cambodia", "Cambodia"],
            "Sex":             ["Total", "Total"],
            "WEALTH_QUINTILE": ["_T", "Q1"],
        })
        result, applied = _resolve_duplicates(df, "TIME_PERIOD", "OBS_VALUE", "Sex")
        assert len(applied) > 0, "Should filter to remove genuine duplicates"
        assert len(result) == 1


# ===========================================================================
# Filter suppression for color_by column
# ===========================================================================

class TestFilterSuppression:
    """The render layer must not apply a filter that would collapse the color_by dimension."""

    def _apply_filters(self, df, filters, color):
        """Replicate the filter logic from _render_chart."""
        color_lower = color.lower() if color else ""
        for col, val in filters.items():
            if col.lower() == color_lower:
                continue
            if col in df.columns:
                filtered = df[df[col] == val]
                if not filtered.empty:
                    df = filtered
        return df

    def test_sex_filter_skipped_when_color_by_sex(self, sample_df_single):
        filters = {"SEX": "_T", "UNIT_MEASURE": "deaths per 1000"}
        result = self._apply_filters(sample_df_single.copy(), filters, "Sex")
        # SEX filter should be skipped; UNIT_MEASURE filter has only one value so no change
        assert len(result) == len(sample_df_single), "Sex rows should not be collapsed"

    def test_other_filters_still_applied(self):
        # Build data where each Sex has both Q1 and _T wealth rows
        df = pd.DataFrame({
            "TIME_PERIOD": [2020] * 6,
            "OBS_VALUE":   [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "Geographic area": ["Cambodia"] * 6,
            "Sex":    ["Male", "Male", "Female", "Female", "Total", "Total"],
            "WEALTH": ["Q1",   "_T",   "Q1",     "_T",    "Q1",    "_T"],
        })
        filters = {"WEALTH": "_T", "SEX": "_T"}   # SEX not a column → skipped
        result = self._apply_filters(df, filters, "Sex")
        assert (result["WEALTH"] == "_T").all(), "WEALTH filter should apply"
        assert set(result["Sex"].unique()) == {"Male", "Female", "Total"}, "All sex groups kept"


# ===========================================================================
# _format_csv_response deduplication
# ===========================================================================

class TestFormatCsvDedup:
    def test_same_geo_indicator_replaces_not_appends(self, isolated_session):
        df1 = pd.DataFrame({"OBS_VALUE": [1], "TIME_PERIOD": [2020],
                             "Geographic area": ["Cambodia"], "INDICATOR": ["X"]})
        df2 = pd.DataFrame({"OBS_VALUE": [2], "TIME_PERIOD": [2021],
                             "Geographic area": ["Cambodia"], "INDICATOR": ["X"]})

        _format_csv_response(df1, "X", "KHM", "http://url1")
        _format_csv_response(df2, "X", "KHM", "http://url2")

        assert len(get_fetched_dfs()) == 1, "Should replace, not append"
        assert get_fetched_dfs()[0]["OBS_VALUE"].iloc[0] == 2

    def test_different_geo_both_kept(self, isolated_session):
        df1 = pd.DataFrame({"OBS_VALUE": [1], "TIME_PERIOD": [2020],
                             "Geographic area": ["Cambodia"], "INDICATOR": ["X"]})
        df2 = pd.DataFrame({"OBS_VALUE": [2], "TIME_PERIOD": [2020],
                             "Geographic area": ["Thailand"], "INDICATOR": ["X"]})

        _format_csv_response(df1, "X", "KHM", "http://url1")
        _format_csv_response(df2, "X", "THA", "http://url2")

        assert len(get_fetched_dfs()) == 2


# ===========================================================================
# get_data_dimensions
# ===========================================================================

class TestGetDataDimensions:
    def test_returns_no_data_message_when_empty(self, isolated_session):
        result = get_data_dimensions()
        assert "No data" in result

    def test_detects_sex_dimension(self, isolated_session, sample_df_single):
        isolated_session.fetched_dfs.append(sample_df_single)
        result = get_data_dimensions()
        assert "Sex" in result or "SEX" in result

    def test_excludes_non_disagg_cols(self, isolated_session, sample_df_single):
        isolated_session.fetched_dfs.append(sample_df_single)
        result = get_data_dimensions()
        assert "OBS_VALUE" not in result
        assert "TIME_PERIOD" not in result
        assert "Geographic area" not in result


# ===========================================================================
# Country resolution
# ===========================================================================

class TestResolveCountry:
    def test_exact_match(self):
        assert _resolve_country("cambodia") == "cambodia"

    def test_case_insensitive(self):
        assert _resolve_country("Cambodia") == "cambodia"

    def test_common_alias(self):
        result = _resolve_country("Laos")
        assert result is not None and "lao" in result

    def test_unknown_country_returns_none(self):
        assert _resolve_country("Wakanda") is None
