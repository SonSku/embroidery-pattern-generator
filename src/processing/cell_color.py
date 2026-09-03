import numpy as np
from skimage.color import rgb2lab, lab2rgb


# calculates average color of physcial pixels inside cell of pixel art
def extract_cell_color(cell: np.ndarray, trim_margin: float = 0.2, outlier_threshold: float = 25.0) -> np.ndarray:
    """
    cell: tablica (block_h, block_w, 3) — piksele jednej logicznej komórki
    trim_margin: jaki % z brzegu komórki pominąć (błędy najczęściej są na styku z sąsiadem)
    outlier_threshold: maksymalna odległość Lab od mediany, żeby piksel się liczył
    """
    h, w, _ = cell.shape
    margin_h = int(h * trim_margin)
    margin_w = int(w * trim_margin)
    # leave margin out in case there is a line or near line artifacts
    inner = cell[margin_h:h-margin_h, margin_w:w-margin_w] if margin_h and margin_w else cell

    pixels = inner.reshape(-1, 3)
    pixels_lab = rgb2lab(pixels)
    pixels_median = np.median(pixels_lab, axis=0) # calculate median color
    distances = np.linalg.norm(pixels_lab - pixels_median, axis=1) # calculate distance of each pixel from median
    mask = distances <= outlier_threshold 
    filtered_lab_pixels = pixels_lab[mask] # in order to avoid artifacts each pixel that diffres a lot from median will be filtered out

    if len(filtered_lab_pixels) == 0:
        filtered_lab_pixels = pixels_lab  # if no pixels were left after filtering just take median

    avg_lab_color = np.average(filtered_lab_pixels, axis=0) # calculate average color
    avg_rgb_color = lab2rgb([[avg_lab_color]])[0, 0]        # Lab to RGB conversion
    avg_color = np.clip(avg_rgb_color * 255, 0, 255).astype(np.uint8)

    return avg_color

# calculates average color of every cell in pixel art
def extract_grid_colors(image: np.ndarray, block_width: int, block_height: int) -> np.ndarray:
    h, w, _ = image.shape
    n_cols = w // block_width
    n_rows = h // block_height

    grid = np.zeros((n_rows, n_cols, 3), dtype=np.uint8)
    for row in range(n_rows):
        for col in range(n_cols):
            cell = image[row*block_height:(row+1)*block_height, col*block_width:(col+1)*block_width]
            grid[row, col] = extract_cell_color(cell)
    return grid
