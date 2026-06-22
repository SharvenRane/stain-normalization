import numpy as np
import pytest

from src.reinhard import ReinhardNormalizer, rgb_to_lab, lab_to_rgb
from tests.synth import synthetic_he_image


def _lab_stats(image_uint8):
    lab = rgb_to_lab(image_uint8.astype(np.float64) / 255.0)
    flat = lab.reshape(-1, 3)
    return flat.mean(axis=0), flat.std(axis=0)


def test_lab_roundtrip_is_close():
    img = synthetic_he_image(seed=3, size=32) / 255.0
    recovered = lab_to_rgb(rgb_to_lab(img))
    assert np.allclose(img, recovered, atol=1e-6)


def test_reinhard_matches_target_lab_statistics():
    target = synthetic_he_image(seed=1, size=64, h_strength=1.0, e_strength=1.0)
    source = synthetic_he_image(
        seed=2, size=64, h_strength=0.4, e_strength=1.4, brightness=0.8
    )

    norm = ReinhardNormalizer().fit(target)
    result = norm.transform(source)

    t_mean, t_std = _lab_stats(target)
    r_mean, r_std = _lab_stats(result)

    # After normalization the LAB channel means and stds should match the target.
    assert np.allclose(r_mean, t_mean, atol=1e-2)
    assert np.allclose(r_std, t_std, atol=1e-2)


def test_reinhard_moves_source_toward_target():
    target = synthetic_he_image(seed=1, size=64)
    source = synthetic_he_image(seed=7, size=64, h_strength=0.3, brightness=0.7)

    t_mean, _ = _lab_stats(target)
    s_mean, _ = _lab_stats(source)

    result = ReinhardNormalizer().fit(target).transform(source)
    r_mean, _ = _lab_stats(result)

    before = np.linalg.norm(s_mean - t_mean)
    after = np.linalg.norm(r_mean - t_mean)
    assert after < before


def test_reinhard_preserves_shape_and_dtype():
    target = synthetic_he_image(seed=1, size=40)
    source = synthetic_he_image(seed=9, size=40)
    result = ReinhardNormalizer().fit(target).transform(source)
    assert result.shape == source.shape
    assert result.dtype == np.uint8


def test_reinhard_requires_fit():
    source = synthetic_he_image(seed=4, size=16)
    with pytest.raises(RuntimeError):
        ReinhardNormalizer().transform(source)


def test_reinhard_rejects_non_rgb():
    norm = ReinhardNormalizer()
    with pytest.raises(ValueError):
        norm.fit(np.zeros((8, 8), dtype=np.uint8))
