"""Reinhard stain normalization.

Reinhard color transfer matches the per channel mean and standard deviation of a
source image to a target image in the LAB color space. For H&E histopathology the
LAB space decorrelates luminance from the two color opponent channels, so shifting
and scaling each channel independently transfers the overall stain appearance of
the target onto the source without large color artifacts.

Reference: E. Reinhard, M. Ashikhmin, B. Gooch, P. Shirley, "Color Transfer between
Images", IEEE Computer Graphics and Applications, 2001.
"""

import numpy as np

# sRGB to LMS-like matrices from Reinhard et al. The forward matrix converts linear
# RGB into a long/medium/short cone response space; LAB is then formed in log space.
_RGB_TO_LMS = np.array(
    [
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444],
    ],
    dtype=np.float64,
)

_LMS_TO_RGB = np.linalg.inv(_RGB_TO_LMS)

# Two stage decomposition of LMS (in log space) into the LAB opponent space.
_LMS_TO_LAB = np.array(
    [
        [1.0 / np.sqrt(3.0), 0.0, 0.0],
        [0.0, 1.0 / np.sqrt(6.0), 0.0],
        [0.0, 0.0, 1.0 / np.sqrt(2.0)],
    ],
    dtype=np.float64,
) @ np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -2.0],
        [1.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

_LAB_TO_LMS = np.linalg.inv(_LMS_TO_LAB)

_EPS = 1e-6


def _to_float(image):
    """Return image as float64 in the range [0, 1] and remember the input dtype."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected an RGB image of shape (H, W, 3).")
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0, np.uint8
    arr = arr.astype(np.float64)
    if arr.max() > 1.0:
        return arr / 255.0, np.float64
    return arr, np.float64


def _from_float(arr, dtype):
    """Clip to [0, 1] and convert back to the requested dtype."""
    arr = np.clip(arr, 0.0, 1.0)
    if dtype == np.uint8:
        return np.round(arr * 255.0).astype(np.uint8)
    return arr.astype(np.float64)


def rgb_to_lab(image01):
    """Convert an RGB image in [0, 1] to the Reinhard LAB space.

    Returns an array of shape (H, W, 3) of float64 LAB values.
    """
    h, w, _ = image01.shape
    flat = image01.reshape(-1, 3)
    lms = flat @ _RGB_TO_LMS.T
    lms = np.clip(lms, _EPS, None)
    log_lms = np.log10(lms)
    lab = log_lms @ _LMS_TO_LAB.T
    return lab.reshape(h, w, 3)


def lab_to_rgb(lab):
    """Inverse of :func:`rgb_to_lab`. Returns an RGB image in [0, 1] (unclipped)."""
    h, w, _ = lab.shape
    flat = lab.reshape(-1, 3)
    log_lms = flat @ _LAB_TO_LMS.T
    lms = np.power(10.0, log_lms)
    rgb = lms @ _LMS_TO_RGB.T
    return rgb.reshape(h, w, 3)


class ReinhardNormalizer:
    """Match the LAB channel statistics of a source image to a fitted target.

    Call :meth:`fit` with the target image, then :meth:`transform` on any source.
    """

    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target):
        """Store the per channel LAB mean and std of the target image."""
        img01, _ = _to_float(target)
        lab = rgb_to_lab(img01)
        flat = lab.reshape(-1, 3)
        self.target_means = flat.mean(axis=0)
        self.target_stds = flat.std(axis=0)
        return self

    def transform(self, source):
        """Normalize ``source`` so its LAB statistics match the target."""
        if self.target_means is None:
            raise RuntimeError("ReinhardNormalizer must be fit before transform.")
        img01, dtype = _to_float(source)
        lab = rgb_to_lab(img01)
        flat = lab.reshape(-1, 3)

        src_means = flat.mean(axis=0)
        src_stds = flat.std(axis=0)
        safe_stds = np.where(src_stds < _EPS, 1.0, src_stds)

        normalized = (flat - src_means) / safe_stds
        normalized = normalized * self.target_stds + self.target_means

        out = lab_to_rgb(normalized.reshape(lab.shape))
        return _from_float(out, dtype)

    def fit_transform(self, target, source):
        return self.fit(target).transform(source)
