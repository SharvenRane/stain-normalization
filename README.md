# stain-normalization

Stain normalization for H&E histopathology images, written in plain NumPy. This repo
implements the two most widely used color normalization methods, Reinhard and
Macenko, and ships with a synthetic image generator so the behavior can be tested
without downloading any slides.

## Why stain normalization

Hematoxylin and eosin slides look different depending on the staining protocol, the
scanner, and the lab. Those color shifts hurt downstream models that were trained on
one site and applied to another. Stain normalization rewrites the colors of a source
image so they match a chosen reference, which removes much of that batch effect while
keeping the tissue structure intact.

## Methods

### Reinhard

The Reinhard method treats normalization as color transfer. It converts the image
into a LAB color space where luminance and the two color opponent channels are close
to uncorrelated, then shifts and scales each channel so its mean and standard
deviation match the reference. The implementation here uses the original Reinhard LMS
to LAB decomposition in log space, so the forward and inverse transforms are exact
up to floating point.

```python
import numpy as np
from src import ReinhardNormalizer

target = ...  # reference RGB image, uint8 (H, W, 3)
source = ...  # image to normalize

norm = ReinhardNormalizer().fit(target)
result = norm.transform(source)
```

After `transform`, the LAB channel means and stds of the result match the target
within a small tolerance. The accompanying test checks exactly that property on a
synthetic image.

### Macenko

The Macenko method is stain aware. It converts every pixel to optical density using
the Beer Lambert law, then finds the two dominant stain directions as the leading
eigenvectors of the optical density cloud. Projecting onto a robust angular range
recovers the hematoxylin and eosin stain vectors, and a least squares solve gives the
per pixel stain concentrations. Normalization rebuilds the image from a reference
stain matrix and a reference concentration scale, so different inputs end up sharing a
common color appearance.

```python
from src import MacenkoNormalizer

# Use the built in reference, or fit a target to learn its stain matrix.
result = MacenkoNormalizer().transform(source)

# Or match a specific reference image:
result = MacenkoNormalizer().fit(target).transform(source)
```

`MacenkoNormalizer` exposes `io`, `alpha`, and `beta`. `io` is the transmitted light
intensity, `alpha` is the percentile used to pick robust stain angle extremes, and
`beta` is the optical density threshold that drops background pixels.

## Layout

```
src/
  reinhard.py    Reinhard LAB color transfer
  macenko.py     Macenko stain separation and normalization
tests/
  synth.py       tiny synthetic H&E image generator
  test_reinhard.py
  test_macenko.py
```

## Image conventions

Both normalizers accept either uint8 RGB arrays in the range 0 to 255 or float arrays
in the range 0 to 1, and they return the same kind of array they were given. Inputs
must have shape `(H, W, 3)`.

## Running the tests

```
pip install -r requirements.txt
pytest tests/ -q
```

The tests build small synthetic H&E images with correlated stain structure, so they
run in a fraction of a second on CPU with no external data. They check real behavior:
the Reinhard output statistics converge to the target, the LAB and optical density
transforms round trip, the Macenko stain vectors come out unit norm and oriented into
positive optical density, and every method preserves image shape and dtype.

On this machine the full suite reports 12 passed.
