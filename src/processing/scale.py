import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb


# resize image to certain width and height; INTER_AREA takes avg color of neighbouring pixels which can help avoid unneccesary artifacts
def downsample_image(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

# reduce the number of colors to n colors
def reduce_colors(image: np.ndarray, n_colors: int) -> np.ndarray:
    h, w, _ = image.shape
    lab = rgb2lab(image / 255.0).reshape(-1, 3) # convert RGB to LAB
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10).fit(lab) # use k-means to search for clusters of most important n colors
    new_lab = kmeans.cluster_centers_[kmeans.labels_].reshape(h, w, 3) # change pixel color to one of n colors
    return (lab2rgb(new_lab) * 255).astype(np.uint8) # convert LAB to RGB