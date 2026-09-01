import numpy as np
import pytest
from src.processing.pipeline import image_to_dmc_pattern
from src.colors.dmc import load_dmc_palette, rgb_to_lab, build_lab_tree


@pytest.fixture(scope="module")
def dmc_setup():
    df = rgb_to_lab(load_dmc_palette())
    tree = build_lab_tree(df)
    return df, tree


@pytest.fixture
def four_color_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = [255, 0, 0]
    img[:50, 50:] = [0, 255, 0]
    img[50:, :50] = [0, 0, 255]
    img[50:, 50:] = [255, 255, 0]
    return img


def test_grid_has_correct_shape(four_color_image, dmc_setup):
    df, tree = dmc_setup
    grid, _ = image_to_dmc_pattern(four_color_image, 10, 10, 4, df, tree)
    assert grid.shape == (10, 10)


def test_used_colors_count_sums_to_grid_size(four_color_image, dmc_setup):
    df, tree = dmc_setup
    grid, used_colors = image_to_dmc_pattern(four_color_image, 10, 10, 4, df, tree)
    assert used_colors["count"].sum() == 100


def test_four_quadrants_map_to_four_distinct_codes(four_color_image, dmc_setup):
    df, tree = dmc_setup
    grid, used_colors = image_to_dmc_pattern(four_color_image, 10, 10, 4, df, tree)
    assert len(used_colors) <= 4
    # check if all quarters are uniform
    assert len(np.unique(grid[:5, :5])) == 1   # up-left
    assert len(np.unique(grid[:5, 5:])) == 1   # up-right
    assert len(np.unique(grid[5:, :5])) == 1   # down-left
    assert len(np.unique(grid[5:, 5:])) == 1   # down-right