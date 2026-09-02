import numpy as np
import matplotlib.pyplot as plt

# converts image colors to grayscale and calculates gradient
# big gradient value means the colors vary a lot on the grayscale hence it might be a line of a pixel
def grayscale_gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = image.mean(axis=2)  # H x W
    grad_x = gray[:, 1:] - gray[:, :-1]
    grad_y = gray[1:, :] - gray[:-1, :]
    return grad_x, grad_y

# sums gradient values along x-axis
# high sum of gradient values across specific axis means new pixel row
def column_profile(grad_x: np.ndarray) -> np.ndarray:
    return np.abs(grad_x).sum(axis=0)

# sums gradient values along y-axis
def row_profile(grad_y: np.ndarray) -> np.ndarray:
    return np.abs(grad_y).sum(axis=1)

# function for testing
def make_synthetic_pixel_art(block_size: int, n_blocks: int) -> np.ndarray:
    blocks = np.random.randint(0, 256, size=(n_blocks, n_blocks, 3), dtype=np.uint8)
    art = np.repeat(blocks, block_size, axis=0)
    art = np.repeat(art, block_size, axis=1)
    
    return art

# FFT decomposes the gradient signal into a sum of sinusoids
# let's say we have a 10x10 pixel art in 300x300 resolution
# that means each color pixel from pixel art is the size of 30x30 physical image pixels (block size = 30)
# the gradient profile spikes up at a certain frequency - when gradient value is high and then drops instantly (color change)
# the distance between peaks is equal to the block size
# in the case of 10x10 pixel in 300x300 resolution the period should be 30
def detect_block_size_fft(profile: np.ndarray) -> int:
    n = len(profile)
    windowed = profile * np.hanning(n)  # reduces leakage on the edges of a signal

    spectrum = np.fft.rfft(windowed)
    magnitude = np.abs(spectrum)
    magnitude[0] = 0

    dominant_idx = np.argmax(magnitude)
    freqs = np.fft.rfftfreq(n, d=1)
    dominant_freq = freqs[dominant_idx]

    if dominant_freq == 0:
        raise ValueError("No period detected")

    period = 1 / dominant_freq
    return round(period)


def detect_grid_size(image: np.ndarray) -> tuple[int, int]:
    grad_x, grad_y = grayscale_gradient(image)
    col_profile = column_profile(grad_x)
    row_profile_vals = row_profile(grad_y)

    block_width = detect_block_size_fft(col_profile)
    block_height = detect_block_size_fft(row_profile_vals)
    return block_width, block_height