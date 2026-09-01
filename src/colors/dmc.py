import pandas as pd
import numpy as np
from skimage.color import rgb2lab # convert RBG to LAB
from scipy.spatial import KDTree # KDTree for finding nearest color


def load_dmc_palette(csv_path: str = "data/colors/DMC_colors.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["code", "name", "hex", "red", "green", "blue"]

    return df

# converts RGB to CIELAB which represents better human vision
def rgb_to_lab(df):
    rgb = df[["red", "green", "blue"]].to_numpy()
    rgb = rgb / 255.0  # Normalize to 0-1 range
    lab = rgb2lab(rgb)
    df["L"], df["a"], df["b"] = lab[:, 0], lab[:, 1], lab[:, 2]
    return df

# creates k-dimensional tree based on  three variables: L, a, b "placing" them on a 3D map
# works quicker than calculating distance between each color separately
def build_lab_tree(df: pd.DataFrame) -> KDTree:
    return KDTree(df[["L", "a", "b"]].to_numpy())

def find_nearest_dmc(rgb: tuple[int, int, int], df: pd.DataFrame, tree: KDTree, k: int = 1):
    lab = rgb2lab(np.array([[rgb]]) / 255.0).reshape(1, 3) # changes variable value from RBG to lab
    distances, indices = tree.query(lab, k=k) # searches for k closest colors
    indices = np.atleast_1d(indices.squeeze())
    return df.iloc[indices]

def main():
    df = load_dmc_palette()
    df = rgb_to_lab(df)
    tree = build_lab_tree(df)

    assert df["code"].is_unique, "WARNING: duplicated DMC colors"

if __name__ == "__main__":
    main()