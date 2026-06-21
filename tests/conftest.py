#
# Copyright (C) 2026 smoother.py.
#
# smoother.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Configuration for the test suite."""

import numpy as np
import pytest


@pytest.fixture
def make_fake_raster():
    """Factory fixture for fake probability rasters."""

    def _make(nclasses, nrows, ncols, seed=42):
        rng = np.random.default_rng(seed)
        raw = rng.random((nclasses, nrows, ncols))

        return raw / raw.sum(axis=0, keepdims=True)

    return _make


@pytest.fixture
def make_dataarray(make_fake_raster):
    """Factory fixture for fake probability DataArrays."""
    import xarray as xr  # noqa: PLC0415

    def _make(nclasses=4, nrows=30, ncols=30, seed=42):
        data = make_fake_raster(nclasses, nrows, ncols, seed)

        return xr.DataArray(
            data,
            dims=("class", "y", "x"),
            coords={
                "class": [f"class_{i}" for i in range(nclasses)],
                "y": np.arange(nrows),
                "x": np.arange(ncols),
            },
            attrs={"units": "probability"},
        )

    return _make
