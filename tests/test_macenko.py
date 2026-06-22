import numpy as np
import pytest

from src.macenko import MacenkoNormalizer, rgb_to_od, od_to_rgb
from tests.synth import synthetic_he_image


def test_od_roundtrip_is_close():
    img = synthetic_he_image(seed=5, size=32).astype(np.float64)
    img = np.maximum(img, 1.0)
    recovered = od_to_rgb(rgb_to_od(img))
    assert np.allclose(img, recovered, atol=1e-3)


def test_macenko_returns_valid_image_same_shape():
    source = synthetic_he_image(seed=2, size=64)
    result = MacenkoNormalizer().transform(source)

    assert result.shape == source.shape
    assert result.dtype == np.uint8
    assert np.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 255


def test_macenko_fit_transform_runs_and_preserves_shape():
    target = synthetic_he_image(seed=1, size=64, h_strength=1.0, e_strength=1.0)
    source = synthetic_he_image(seed=3, size=48, h_strength=0.5, brightness=0.7)

    norm = MacenkoNormalizer()
    result = norm.fit(target).transform(source)

    assert result.shape == source.shape
    assert result.dtype == np.uint8
    assert np.isfinite(result).all()


def test_macenko_estimates_unit_norm_stain_vectors():
    target = synthetic_he_image(seed=1, size=80)
    norm = MacenkoNormalizer()
    img255 = target.astype(np.float64)
    stains = norm._stain_matrix(img255)

    assert stains.shape == (2, 3)
    norms = np.linalg.norm(stains, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    # Stain OD vectors should point into positive optical density overall.
    assert np.all(stains.sum(axis=1) > 0)


def test_macenko_float_input_returns_float01():
    source = (synthetic_he_image(seed=6, size=40).astype(np.float64) / 255.0)
    result = MacenkoNormalizer().transform(source)
    assert result.dtype == np.float64
    assert result.min() >= 0.0 and result.max() <= 1.0
    assert result.shape == source.shape


def test_macenko_rejects_non_rgb():
    norm = MacenkoNormalizer()
    with pytest.raises(ValueError):
        norm.transform(np.zeros((8, 8, 1), dtype=np.uint8))
