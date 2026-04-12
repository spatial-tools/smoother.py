#
# Copyright (C) 2024 smoother.py.
#
# smoother.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Test Bayesian smoothing with synthetic rasters."""

import numpy as np

import smoother


def _make_fake_raster(nclasses, nrows, ncols, seed=42):
    """Generate a fake probability raster that sums to 1 per pixel."""
    rng = np.random.default_rng(seed)
    raw = rng.random((nclasses, nrows, ncols))

    return raw / raw.sum(axis=0, keepdims=True)


def test_output_shape():
    """Output shape must match input shape."""
    data = _make_fake_raster(nclasses=4, nrows=50, ncols=50)
    result = smoother.bayes(data)

    assert result.shape == data.shape


def test_output_range():
    """All values should be valid probabilities in [0, 1]."""
    data = _make_fake_raster(nclasses=5, nrows=40, ncols=40)

    result = smoother.bayes(data)

    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_uniform_input_stays_uniform():
    """Uniform probabilities should remain roughly uniform after smoothing."""
    nclasses = 3
    data = np.full((nclasses, 20, 20), 1.0 / nclasses)

    result = smoother.bayes(data)

    np.testing.assert_allclose(result, 1.0 / nclasses, atol=1e-5)


def test_custom_parameters():
    """Verify smoothing runs with non-default parameters."""
    data = _make_fake_raster(nclasses=3, nrows=25, ncols=25)

    result = smoother.bayes(data, window_size=5, smoothness=20, neigh_fraction=0.3)

    assert result.shape == data.shape
    assert np.all(np.isfinite(result))
