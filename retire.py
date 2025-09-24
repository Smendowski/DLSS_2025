import numpy as np
from PIL import Image, ImageDraw
from sklearn.base import TransformerMixin


def transform_from_spherical_to_cartesian(s_coords):
    r_v = s_coords[:, 0]
    theta_m = s_coords[:, 1:]
    cos_m = np.cos(theta_m).T
    sin_m = np.sin(theta_m).T

    x_v = []
    for _ in range(0, len(cos_m)):
        x_v.append(r_v * np.prod(sin_m[:_], axis=0) * cos_m[_])

    x_v.append(r_v * np.prod(sin_m, axis=0))

    return np.stack(x_v).T


class PACMAN(TransformerMixin):
    def __init__(self, size=100, scaling='feature', range=(0.1, 0.9)):
        self.size = size
        self.scaling = scaling
        self.range = range

    def vec_to_radar(self, vec):
        n_features = len(vec)

        if self.scaling == "sample":
            vec = (vec - np.min(vec)) / (np.max(vec) - np.min(vec))
        else:
            X_std = (vec - self._vmin) / (self._vmax - self._vmin)
            vec = X_std * (self.range[1] - self.range[0]) + self.range[0]
            # vec = 0.2 + np.nan_to_num((vec - self._vmin) / (self._vmax - self._vmin))
        scoords = np.stack([
            vec,
            np.linspace(0, 2 * np.pi, n_features, endpoint=False)
        ]).T

        outline = np.stack([
            np.array([.9 for i in range(vec.shape[0])]),
            np.linspace(0, 2 * np.pi, n_features, endpoint=False)
        ]).T

        xy = transform_from_spherical_to_cartesian(scoords) + 1
        xy *= self.size / 2

        outline_xy = transform_from_spherical_to_cartesian(outline) + 1
        outline_xy *= self.size / 2

        img = Image.new("RGB", (self.size, self.size), "#000000")
        ImageDraw.Draw(img).polygon(list(map(tuple, xy)), fill="#ffffff")
        ImageDraw.Draw(img).polygon(list(map(tuple, outline_xy)), fill=None)
        return np.array(img)

    def fit(self, X, y=None):
        if self.scaling == 'feature':
            self._vmax = np.max(X, axis=0)
            self._vmin = np.min(X, axis=0)
        if self.scaling == 'global':
            self._vmax = np.max(X)
            self._vmin = np.min(X)
        return self

    def transform(self, X):
        return np.array([self.vec_to_radar(x) for x in X])
