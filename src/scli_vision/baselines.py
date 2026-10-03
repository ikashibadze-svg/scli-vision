from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import transform
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    return (0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]).astype(np.float32)


class HOGBaseline:
    def __init__(self, image_size: int = 64):
        self.image_size = image_size
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=2.0, random_state=17),
        )

    def _feature(self, image: np.ndarray) -> np.ndarray:
        g = transform.resize(_gray(image), (self.image_size, self.image_size), anti_aliasing=True)
        return hog(
            g,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        ).astype(np.float32)

    def fit(self, images: list[np.ndarray], labels: np.ndarray) -> "HOGBaseline":
        X = np.asarray([self._feature(x) for x in images], dtype=np.float32)
        self.model.fit(X, labels)
        return self

    def predict(self, images: list[np.ndarray]) -> np.ndarray:
        X = np.asarray([self._feature(x) for x in images], dtype=np.float32)
        return self.model.predict(X)
