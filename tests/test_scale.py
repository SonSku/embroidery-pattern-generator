import numpy as np
import pytest
from src.processing.scale import downsample_image, reduce_colors


@pytest.fixture
def four_color_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = [255, 0, 0]
    img[:50, 50:] = [0, 255, 0]
    img[50:, :50] = [0, 0, 255]
    img[50:, 50:] = [255, 255, 0]
    return img


def test_downsample_returns_correct_shape(four_color_image):
    result = downsample_image(four_color_image, target_width=10, target_height=15)
    assert result.shape == (15, 10, 3)


def test_downsample_preserves_dtype(four_color_image):
    result = downsample_image(four_color_image, target_width=10, target_height=10)
    assert result.dtype == np.uint8


def test_reduce_colors_respects_max_color_count(four_color_image):
    small = downsample_image(four_color_image, target_width=20, target_height=20)
    reduced = reduce_colors(small, n_colors=4)
    unique_colors = np.unique(reduced.reshape(-1, 3), axis=0)
    assert len(unique_colors) <= 4


def test_reduce_colors_output_shape_matches_input(four_color_image):
    small = downsample_image(four_color_image, target_width=20, target_height=20)
    reduced = reduce_colors(small, n_colors=4)
    assert reduced.shape == small.shape


def test_reduce_colors_single_color_image_stays_uniform():
    solid = np.full((10, 10, 3), [128, 64, 200], dtype=np.uint8)
    reduced = reduce_colors(solid, n_colors=3)
    unique_colors = np.unique(reduced.reshape(-1, 3), axis=0)
    assert len(unique_colors) == 1