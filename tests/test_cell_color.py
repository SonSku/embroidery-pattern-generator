import numpy as np
import pytest

from src.processing.cell_color import extract_grid_colors, extract_cell_color


def test_extract_cell_color_uniform_cell_returns_that_color():
    cell = np.full((10, 10, 3), [200, 50, 50], dtype=np.uint8)
    result = extract_cell_color(cell)
    assert np.allclose(result, [200, 50, 50], atol=3)


def test_extract_cell_color_returns_uint8_in_valid_range():
    cell = np.full((10, 10, 3), [10, 240, 128], dtype=np.uint8)
    result = extract_cell_color(cell)
    assert result.dtype == np.uint8
    assert np.all(result >= 0) and np.all(result <= 255)


def test_extract_cell_color_ignores_outlier_pixel_in_center():
    # artefakt w środku komórki, poza marginesem trim_margin, więc musi zostać
    # odrzucony przez mechanizm outlierów, nie przez trimming
    cell = np.full((10, 10, 3), [200, 50, 50], dtype=np.uint8)
    cell[5, 5] = [0, 255, 0]
    result = extract_cell_color(cell)
    assert np.allclose(result, [200, 50, 50], atol=5)


def test_extract_cell_color_ignores_edge_artifacts_via_trim_margin():
    # artefakty tylko na samym brzegu komórki - trim_margin powinien je wyciąć
    cell = np.full((10, 10, 3), [80, 80, 200], dtype=np.uint8)
    cell[0, :] = [255, 255, 255]
    cell[-1, :] = [255, 255, 255]
    cell[:, 0] = [255, 255, 255]
    cell[:, -1] = [255, 255, 255]
    result = extract_cell_color(cell)
    assert np.allclose(result, [80, 80, 200], atol=5)


def test_extract_cell_color_two_distinct_colors_are_not_averaged_blindly():
    # połowa komórki jednym kolorem, połowa drugim - upewnij się, że wynik
    # nie jest po prostu "gdzieś pomiędzy", tylko realnym, sensownym kolorem
    # (test dokumentujący zachowanie, nie sprawdzający jednej "poprawnej" wartości)
    cell = np.zeros((10, 10, 3), dtype=np.uint8)
    cell[:, :5] = [255, 0, 0]
    cell[:, 5:] = [0, 0, 255]
    result = extract_cell_color(cell)
    assert result.shape == (3,)
    assert result.dtype == np.uint8


def test_extract_cell_color_handles_all_outliers_without_crashing():
    # bardzo niski threshold odrzuci prawie wszystkie piksele - funkcja
    # nie powinna się wysypać (fallback albo świadomie obsłużony przypadek)
    cell = np.random.default_rng(0).integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
    result = extract_cell_color(cell, outlier_threshold=0.0001)
    assert result.shape == (3,)
    assert result.dtype == np.uint8


def test_extract_cell_color_no_trim_margin_still_works():
    cell = np.full((6, 6, 3), [120, 30, 200], dtype=np.uint8)
    result = extract_cell_color(cell, trim_margin=0.0)
    assert np.allclose(result, [120, 30, 200], atol=3)


# --- extract_grid_colors ---

def test_extract_grid_colors_returns_correct_shape():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    grid = extract_grid_colors(image, block_width=10, block_height=10)
    assert grid.shape == (4, 6, 3)  # n_rows=40/10, n_cols=60/10


def test_extract_grid_colors_recovers_four_quadrant_colors():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:10, :10] = [255, 0, 0]
    image[:10, 10:] = [0, 255, 0]
    image[10:, :10] = [0, 0, 255]
    image[10:, 10:] = [255, 255, 0]

    grid = extract_grid_colors(image, block_width=10, block_height=10)

    assert grid.shape == (2, 2, 3)
    assert np.allclose(grid[0, 0], [255, 0, 0], atol=3)
    assert np.allclose(grid[0, 1], [0, 255, 0], atol=3)
    assert np.allclose(grid[1, 0], [0, 0, 255], atol=3)
    assert np.allclose(grid[1, 1], [255, 255, 0], atol=3)


def test_extract_grid_colors_ignores_incomplete_trailing_blocks():
    # obraz nie jest idealną wielokrotnością block_size - nadmiarowe piksele
    # na brzegu powinny zostać po prostu pominięte, bez błędu
    image = np.zeros((23, 25, 3), dtype=np.uint8)
    grid = extract_grid_colors(image, block_width=10, block_height=10)
    assert grid.shape == (2, 2, 3)  # 23//10=2, 25//10=2


def test_extract_grid_colors_dtype_is_uint8():
    image = np.full((20, 20, 3), [10, 200, 90], dtype=np.uint8)
    grid = extract_grid_colors(image, block_width=10, block_height=10)
    assert grid.dtype == np.uint8
