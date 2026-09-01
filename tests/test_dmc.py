import pytest
import pandas as pd
from src.colors.dmc import load_dmc_palette, rgb_to_lab, build_lab_tree, find_nearest_dmc


@pytest.fixture
def dmc_df():
    df = load_dmc_palette()
    return rgb_to_lab(df)


@pytest.fixture
def dmc_tree(dmc_df):
    return build_lab_tree(dmc_df)


def test_load_dmc_palette_has_expected_columns():
    df = load_dmc_palette()
    assert list(df.columns) == ["code", "name", "hex", "red", "green", "blue"]


def test_load_dmc_palette_no_duplicates():
    df = load_dmc_palette()
    assert df["code"].is_unique


def test_load_dmc_palette_no_nulls():
    df = load_dmc_palette()
    assert df.isna().sum().sum() == 0


def test_rgb_to_lab_adds_columns(dmc_df):
    assert {"L", "a", "b"}.issubset(dmc_df.columns)


def test_rgb_to_lab_black_has_low_lightness(dmc_df):
    # black (0,0,0) should have L near 0
    black_row = dmc_df[dmc_df["hex"] == "#000000"]
    assert not black_row.empty
    assert black_row["L"].iloc[0] < 5


def test_find_nearest_dmc_returns_correct_count(dmc_df, dmc_tree):
    result = find_nearest_dmc((0, 0, 0), dmc_df, dmc_tree, k=3)
    assert len(result) == 3


def test_find_nearest_dmc_black_matches_black(dmc_df, dmc_tree):
    result = find_nearest_dmc((0, 0, 0), dmc_df, dmc_tree, k=1)
    assert result["name"].iloc[0] == "Black"


def test_find_nearest_dmc_white_matches_white(dmc_df, dmc_tree):
    result = find_nearest_dmc((255, 255, 255), dmc_df, dmc_tree, k=1)
    assert "White" in result["name"].iloc[0] or "white" in result["name"].iloc[0].lower()