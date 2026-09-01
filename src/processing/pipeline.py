import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from src.processing.scale import downsample_image, reduce_colors
from src.colors.dmc import find_nearest_dmc


# converts image into a grid of set height and width with DMC codes of n colors
# aside from grid this function returns DataFrame with list of used DMC colors and how many times they appear
def image_to_dmc_pattern(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    n_colors: int,
    dmc_df: pd.DataFrame,
    dmc_tree: KDTree,
) -> tuple[np.ndarray, pd.DataFrame]:
    small = downsample_image(image, target_width, target_height) # change image size
    reduced = reduce_colors(small, n_colors) # reduce colors

    grid_codes = np.empty((target_height, target_width), dtype=object) # create pattern grid
    for y in range(target_height):
        for x in range(target_width):
            rgb = tuple(reduced[y, x])
            nearest = find_nearest_dmc(rgb, dmc_df, dmc_tree, k=1)
            grid_codes[y, x] = nearest["code"].iloc[0] # assign nearest DMC code

    codes, counts = np.unique(grid_codes, return_counts=True) # check how many times each of the color have been used
    used_colors = dmc_df[dmc_df["code"].isin(codes)].copy()
    used_colors["count"] = used_colors["code"].map(dict(zip(codes, counts))) # create DataFrame with colors and their usage

    return grid_codes, used_colors


def main():
    from src.colors.dmc import load_dmc_palette, rgb_to_lab, build_lab_tree

    dmc_df = load_dmc_palette()
    dmc_df = rgb_to_lab(dmc_df)
    dmc_tree = build_lab_tree(dmc_df)

    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_img[:50, :50] = [255, 0, 0]
    test_img[:50, 50:] = [0, 255, 0]
    test_img[50:, :50] = [0, 0, 255]
    test_img[50:, 50:] = [255, 255, 0]

    grid, used_colors = image_to_dmc_pattern(
        test_img, target_width=10, target_height=10, n_colors=4,
        dmc_df=dmc_df, dmc_tree=dmc_tree,
    )
    print("grid shape:", grid.shape)
    print("grid:\n", grid)
    print("\nused colors:\n", used_colors[["code", "name", "count"]])


if __name__ == "__main__":
    main()