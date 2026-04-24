"""
Shared pytest fixtures and helpers.
"""
import pandas as pd
import pytest

from agents.sdmx_agent import AgentSession, set_current_session


@pytest.fixture(autouse=True)
def isolated_session():
    """Give every test its own AgentSession so state never leaks between tests."""
    session = AgentSession()
    set_current_session(session)
    yield session


@pytest.fixture
def sample_df_single():
    """Single-country, single-indicator DataFrame with a Sex dimension."""
    return pd.DataFrame({
        "TIME_PERIOD":    [2019, 2020, 2021, 2019, 2020, 2021, 2019, 2020, 2021],
        "OBS_VALUE":      [10.1, 10.5, 10.9, 9.8, 10.2, 10.7, 20.0, 20.7, 21.6],
        "Geographic area":["Cambodia"] * 9,
        "Sex":            ["Male"] * 3 + ["Female"] * 3 + ["Total"] * 3,
        "SEX":            ["M"] * 3 + ["F"] * 3 + ["_T"] * 3,
        "INDICATOR":      ["CME_MRY0T4"] * 9,
        "UNIT_MEASURE":   ["deaths per 1000"] * 9,
    })


@pytest.fixture
def sample_df_multi():
    """Multi-country DataFrame (Cambodia, Thailand, Vietnam) with Sex dimension."""
    countries = ["Cambodia", "Thailand", "Vietnam"]
    rows = []
    for country in countries:
        for year in [2019, 2020, 2021]:
            for sex, sex_code in [("Male", "M"), ("Female", "F"), ("Total", "_T")]:
                rows.append({
                    "TIME_PERIOD":     year,
                    "OBS_VALUE":       float(hash(country + str(year) + sex) % 50 + 10),
                    "Geographic area": country,
                    "Sex":             sex,
                    "SEX":             sex_code,
                    "INDICATOR":       "CME_MRY0T4",
                    "UNIT_MEASURE":    "deaths per 1000",
                })
    return pd.DataFrame(rows)
