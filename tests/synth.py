"""Helpers to build small synthetic H&E like images for tests."""

import numpy as np

# Approximate RGB appearance of hematoxylin (purple/blue) and eosin (pink) stains.
_H_RGB = np.array([110.0, 80.0, 170.0])
_E_RGB = np.array([210.0, 120.0, 170.0])


def synthetic_he_image(seed=0, size=48, h_strength=1.0, e_strength=1.0, brightness=1.0):
    """Render a tiny synthetic H&E image as a uint8 RGB array.

    Two random concentration fields modulate a hematoxylin and an eosin color so the
    result has realistic correlated color structure. ``h_strength``, ``e_strength``
    and ``brightness`` let a caller make a clearly different looking "source" image.
    """
    rng = np.random.default_rng(seed)
    h_conc = rng.uniform(0.0, 1.0, size=(size, size)) * h_strength
    e_conc = rng.uniform(0.0, 1.0, size=(size, size)) * e_strength

    white = np.array([255.0, 255.0, 255.0])
    # Multiplicative stain absorption against a white background.
    img = white.copy()[None, None, :] * np.ones((size, size, 1))
    img = img * (1.0 - 0.6 * h_conc[..., None] * (1.0 - _H_RGB / 255.0))
    img = img * (1.0 - 0.6 * e_conc[..., None] * (1.0 - _E_RGB / 255.0))
    img = img * brightness

    return np.clip(img, 0, 255).astype(np.uint8)
