"""Macenko stain normalization.

The Macenko method separates an H&E image into its underlying stain concentrations
using the Beer Lambert law. Each pixel is converted to optical density (OD), the two
dominant stain vectors are recovered as the leading singular vectors of the OD cloud,
and the per pixel stain concentrations are obtained by projecting OD onto those
vectors. Normalization then rebuilds the image using a reference stain matrix and a
reference concentration scale, which makes images from different scanners or labs
share a common color appearance.

Reference: M. Macenko et al., "A method for normalizing histology slides for
quantitative analysis", IEEE ISBI, 2009.
"""

import numpy as np

# Reference H&E stain matrix (rows are hematoxylin and eosin OD vectors) and the
# reference maximum stain concentrations, taken from the standard Macenko reference.
_DEFAULT_STAIN_MATRIX = np.array(
    [
        [0.5626, 0.7201, 0.4062],
        [0.2159, 0.8012, 0.5581],
    ],
    dtype=np.float64,
)

_DEFAULT_MAX_C = np.array([1.9705, 1.0308], dtype=np.float64)

_EPS = 1e-6


def _to_float(image):
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected an RGB image of shape (H, W, 3).")
    if arr.dtype == np.uint8:
        return arr.astype(np.float64), np.uint8
    arr = arr.astype(np.float64)
    if arr.max() <= 1.0:
        return arr * 255.0, np.float64
    return arr, np.float64


def _from_float(arr, dtype):
    arr = np.clip(arr, 0.0, 255.0)
    if dtype == np.uint8:
        return np.round(arr).astype(np.uint8)
    return (arr / 255.0).astype(np.float64)


def rgb_to_od(image255, io=240.0):
    """Convert an RGB image in [0, 255] to optical density.

    OD = -log10(I / Io). Intensities are floored to 1 to avoid log of zero.
    """
    img = np.maximum(image255.astype(np.float64), 1.0)
    return -np.log10(img / io)


def od_to_rgb(od, io=240.0):
    """Inverse of :func:`rgb_to_od`, returning RGB in [0, 255] (unclipped)."""
    return io * np.power(10.0, -od)


class MacenkoNormalizer:
    """Macenko stain normalizer.

    Parameters
    ----------
    io : float
        Transmitted light intensity used in the Beer Lambert conversion.
    alpha : float
        Percentile (and its complement) used to find robust stain angle extremes.
    beta : float
        OD threshold below which pixels are treated as background and dropped.
    """

    def __init__(self, io=240.0, alpha=1.0, beta=0.15):
        self.io = float(io)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.target_stain_matrix = _DEFAULT_STAIN_MATRIX.copy()
        self.target_max_c = _DEFAULT_MAX_C.copy()

    def _stain_matrix(self, image255):
        """Estimate the 2x3 H&E stain matrix from an image in [0, 255]."""
        od = rgb_to_od(image255, self.io).reshape(-1, 3)
        mask = np.all(od > self.beta, axis=1)
        od_hat = od[mask]
        if od_hat.shape[0] < 2:
            # Degenerate (near uniform) image: fall back to the reference matrix.
            return self.target_stain_matrix.copy()

        cov = np.cov(od_hat, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        # Two eigenvectors with the largest eigenvalues span the stain plane.
        plane = eig_vecs[:, [-1, -2]]

        proj = od_hat @ plane
        angles = np.arctan2(proj[:, 1], proj[:, 0])

        min_angle = np.percentile(angles, self.alpha)
        max_angle = np.percentile(angles, 100.0 - self.alpha)

        v_min = plane @ np.array([np.cos(min_angle), np.sin(min_angle)])
        v_max = plane @ np.array([np.cos(max_angle), np.sin(max_angle)])

        # Order so that hematoxylin (the first stain) comes first by convention.
        if v_min[0] > v_max[0]:
            stains = np.array([v_min, v_max])
        else:
            stains = np.array([v_max, v_min])

        # Sign convention: stain vectors point into positive OD.
        stains = np.where(stains.sum(axis=1, keepdims=True) < 0, -stains, stains)
        norms = np.linalg.norm(stains, axis=1, keepdims=True)
        norms = np.where(norms < _EPS, 1.0, norms)
        return stains / norms

    def _concentrations(self, image255, stain_matrix):
        """Solve OD = C @ stain_matrix for the per pixel concentrations C."""
        od = rgb_to_od(image255, self.io).reshape(-1, 3)
        # Least squares solve: C has shape (n_pixels, 2).
        concentrations, *_ = np.linalg.lstsq(stain_matrix.T, od.T, rcond=None)
        return concentrations.T

    def fit(self, target):
        """Estimate and store the target stain matrix and concentration scale."""
        img255, _ = _to_float(target)
        stain_matrix = self._stain_matrix(img255)
        conc = self._concentrations(img255, stain_matrix)
        max_c = np.percentile(conc, 99.0, axis=0)
        max_c = np.where(max_c < _EPS, 1.0, max_c)
        self.target_stain_matrix = stain_matrix
        self.target_max_c = max_c
        return self

    def transform(self, source):
        """Normalize ``source`` onto the stored (or default) target stain space."""
        img255, dtype = _to_float(source)
        h, w, _ = img255.shape

        stain_matrix = self._stain_matrix(img255)
        conc = self._concentrations(img255, stain_matrix)

        src_max_c = np.percentile(conc, 99.0, axis=0)
        src_max_c = np.where(src_max_c < _EPS, 1.0, src_max_c)

        # Rescale source concentrations to the target dynamic range.
        conc = conc / src_max_c * self.target_max_c

        od = conc @ self.target_stain_matrix
        rgb = od_to_rgb(od, self.io).reshape(h, w, 3)
        return _from_float(rgb, dtype)

    def fit_transform(self, target, source):
        return self.fit(target).transform(source)
