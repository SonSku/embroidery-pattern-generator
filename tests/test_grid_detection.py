import numpy as np
import pytest

from src.processing.grid_detection import (
    grayscale_gradient,
    column_profile,
    row_profile,
    make_synthetic_pixel_art,
    detect_block_size_fft,
    detect_grid_size,
)


def make_block_image(block_width: int, block_height: int, n_cols: int, n_rows: int, seed: int = 0) -> np.ndarray:
    """Deterministic pixel-art-like image with possibly different x/y block sizes."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(n_rows, n_cols, 3), dtype=np.uint8)
    art = np.repeat(blocks, block_height, axis=0)
    art = np.repeat(art, block_width, axis=1)
    return art


def test_grayscale_gradient_shapes():
    img = make_block_image(block_width=5, block_height=5, n_cols=4, n_rows=4)
    grad_x, grad_y = grayscale_gradient(img)
    h, w, _ = img.shape
    assert grad_x.shape == (h, w - 1)
    assert grad_y.shape == (h - 1, w)


def test_grayscale_gradient_zero_for_uniform_image():
    img = np.full((20, 20, 3), 100, dtype=np.uint8)
    grad_x, grad_y = grayscale_gradient(img)
    assert np.all(grad_x == 0)
    assert np.all(grad_y == 0)


def test_grayscale_gradient_detects_single_edge():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, 5:] = 255
    grad_x, _ = grayscale_gradient(img)
    # the only nonzero column-wise difference should be at the boundary (col index 4 -> 5)
    nonzero_cols = np.nonzero(grad_x.sum(axis=0))[0]
    assert list(nonzero_cols) == [4]


def test_column_profile_length_matches_grad_x_width():
    img = make_block_image(block_width=6, block_height=6, n_cols=5, n_rows=5)
    grad_x, _ = grayscale_gradient(img)
    profile = column_profile(grad_x)
    assert profile.shape == (grad_x.shape[1],)


def test_row_profile_length_matches_grad_y_height():
    img = make_block_image(block_width=6, block_height=6, n_cols=5, n_rows=5)
    _, grad_y = grayscale_gradient(img)
    profile = row_profile(grad_y)
    assert profile.shape == (grad_y.shape[0],)


def test_profiles_are_nonnegative():
    img = make_block_image(block_width=4, block_height=4, n_cols=6, n_rows=6)
    grad_x, grad_y = grayscale_gradient(img)
    assert np.all(column_profile(grad_x) >= 0)
    assert np.all(row_profile(grad_y) >= 0)


def test_detect_block_size_fft_recovers_known_period():
    period = 12
    n_periods = 8
    n = period * n_periods
    x = np.arange(n)
    # clean periodic spike train mimicking a gradient profile at block edges
    profile = (x % period == 0).astype(float) * 50.0
    detected = detect_block_size_fft(profile)
    assert abs(detected - period) <= 1


def test_detect_block_size_fft_raises_on_flat_profile():
    profile = np.zeros(64)
    with pytest.raises(ValueError):
        detect_block_size_fft(profile)


@pytest.mark.parametrize("block_size,n_blocks", [(10, 8), (6, 12), (20, 5)])
def test_detect_grid_size_square_blocks(block_size, n_blocks):
    img = make_block_image(
        block_width=block_size, block_height=block_size,
        n_cols=n_blocks, n_rows=n_blocks, seed=42,
    )
    block_width, block_height = detect_grid_size(img)
    assert abs(block_width - block_size) <= 1
    assert abs(block_height - block_size) <= 1


def test_detect_grid_size_rectangular_blocks():
    img = make_block_image(block_width=6, block_height=9, n_cols=10, n_rows=8, seed=1)
    block_width, block_height = detect_grid_size(img)
    assert abs(block_width - 6) <= 1
    assert abs(block_height - 9) <= 1


def test_detect_grid_size_matches_make_synthetic_pixel_art():
    np.random.seed(7)
    block_size = 8
    img = make_synthetic_pixel_art(block_size=block_size, n_blocks=10)
    block_width, block_height = detect_grid_size(img)
    assert abs(block_width - block_size) <= 1
    assert abs(block_height - block_size) <= 1