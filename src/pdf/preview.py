import numpy as np
import pandas as pd
import cv2


# renders a grid of DMC colors as upscaled RGB image to create preview
def render_preview_image(grid_codes: np.ndarray, dmc_df: pd.DataFrame, cell_pixels: int = 20) -> np.ndarray:
    code_to_rgb = dict(zip(dmc_df["code"], zip(dmc_df["red"], dmc_df["green"], dmc_df["blue"])))

    # create empty grid
    n_rows, n_cols = grid_codes.shape
    preview = np.zeros((n_rows, n_cols, 3), dtype=np.uint8)

    # filling the grid with colors
    for row in range(n_rows):
        for col in range(n_cols):
            preview[row, col] = code_to_rgb[grid_codes[row, col]]

    # upscale the image 
    upscaled = cv2.resize(
        preview, (n_cols * cell_pixels, n_rows * cell_pixels),
        interpolation=cv2.INTER_NEAREST, # INTER_NEAREST so the zoom won't blur pixels
    )
    # cv2 works in BGR so conversion to RGB is needed
    return cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)

# same as render_preview_image but returns RGB instead of BGR - needed for reportlab, not cv2.imwrite
def render_preview_image_rgb(grid_codes: np.ndarray, dmc_df: pd.DataFrame, cell_pixels: int = 20) -> np.ndarray:
    bgr = render_preview_image(grid_codes, dmc_df, cell_pixels)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def main():
    import cv2
    from src.colors.dmc import load_dmc_palette, rgb_to_lab, build_lab_tree
    from src.processing.pipeline import image_to_dmc_pattern, image_to_dmc_pattern_pixelart

    dmc_df = load_dmc_palette()
    dmc_df = rgb_to_lab(dmc_df)
    dmc_tree = build_lab_tree(dmc_df)

    real_photo = cv2.imread("test_images/scaled.png")
    real_photo = cv2.cvtColor(real_photo, cv2.COLOR_BGR2RGB)

    grid_scaled, used_scaled = image_to_dmc_pattern(
        real_photo, target_width=60, target_height=60, n_colors=4,
        dmc_df=dmc_df, dmc_tree=dmc_tree,
    )
    preview_scaled = render_preview_image(grid_scaled, dmc_df)
    cv2.imwrite("preview_scaled.png", preview_scaled)
    print("Tryb skalowania: siatka", grid_scaled.shape, ", kolorów:", len(used_scaled))
    
    pixel_art_img = cv2.imread("test_images/pixelart.png")
    pixel_art_img = cv2.cvtColor(pixel_art_img, cv2.COLOR_BGR2RGB)

    grid_px, used_px = image_to_dmc_pattern_pixelart(pixel_art_img, dmc_df=dmc_df, dmc_tree=dmc_tree)
    preview_px = render_preview_image(grid_px, dmc_df)
    cv2.imwrite("preview_pixelart.png", preview_px)
    print("Tryb pixel art: siatka", grid_px.shape, ", kolorów:", len(used_px))


if __name__ == "__main__":
    main()