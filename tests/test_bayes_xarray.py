#
# Copyright (C) 2026 smoother.py.
#
# smoother.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Test Bayesian smoothing with xarray."""

import numpy as np
import pytest

import smoother

xr = pytest.importorskip("xarray")


def test_dataarray_matches_numpy(make_dataarray):
    """DataArray results must equal the numpy backend results."""
    da = make_dataarray()

    result = smoother.smooth(da)
    expected = smoother.smooth(da.values)

    np.testing.assert_array_equal(result.values, expected)


def test_metadata_preserved(make_dataarray):
    """Dims, coords and attrs must survive the smoothing."""
    da = make_dataarray()

    result = smoother.smooth(da)

    assert result.dims == da.dims
    assert result.attrs == da.attrs
    assert result.coords["class"].values.tolist() == da.coords["class"].values.tolist()


def test_multi_layer(make_fake_raster):
    """Extra dims (e.g. time) are smoothed once per layer."""
    layers = [make_fake_raster(3, 20, 20, seed=seed) for seed in (1, 2, 3)]

    # create cube
    cube = xr.DataArray(np.stack(layers), dims=("time", "class", "y", "x"))

    # smooth!
    result = smoother.smooth(cube)

    assert result.dims == cube.dims

    # test each layer
    for t, layer in enumerate(layers):
        np.testing.assert_array_equal(
            result.isel(time=t).values, smoother.smooth(layer)
        )


def test_dim_inference_by_name(make_fake_raster):
    """Recognized dim names are found regardless of their order."""
    data = make_fake_raster(3, 20, 25)
    da = xr.DataArray(np.moveaxis(data, 0, -1), dims=("latitude", "longitude", "band"))

    # smooth
    result = smoother.smooth(da)
    expected = smoother.smooth(data)

    # assert
    assert result.dims == da.dims
    np.testing.assert_array_equal(np.moveaxis(result.values, -1, 0), expected)


def test_dim_positional_fallback_and_explicit(make_fake_raster):
    """Unknown 3D dims fall back to (class, y, x) order. Explicit names win."""
    data = make_fake_raster(3, 20, 20)
    da = xr.DataArray(data, dims=("c", "r", "q"))
    expected = smoother.smooth(data)

    # smooth
    result_positional = smoother.smooth(data=da)

    # smooth with dim names
    result_explicit = smoother.smooth(data=da, class_dim="c", y_dim="r", x_dim="q")

    # test!
    np.testing.assert_array_equal(result_positional.values, expected)
    np.testing.assert_array_equal(result_explicit.values, expected)


def test_unresolvable_dims_raise(make_fake_raster):
    """A >3D input with unrecognizable dim names must raise an error."""
    data = np.stack([make_fake_raster(3, 10, 10, seed=seed) for seed in (1, 2)])

    cube = xr.DataArray(data, dims=("t", "c", "r", "q"))

    with pytest.raises(ValueError, match="class_dim"):
        smoother.smooth(cube)


def test_invalid_explicit_dim_raises(make_dataarray):
    """An explicit dim name absent from the input must raise an error."""
    da = make_dataarray()

    with pytest.raises(ValueError, match="y_dim"):
        smoother.smooth(da, y_dim="row")


def test_numpy_rejects_dim_params(make_fake_raster):
    """The numpy backend has no dim parameters, so it must raise an error."""
    data = make_fake_raster(3, 10, 10)

    with pytest.raises(TypeError):
        smoother.smooth(data, class_dim="class")


def test_unsupported_type():
    """Unsupported input types must raise an error."""
    with pytest.raises(TypeError, match="does not support"):
        smoother.smooth([[0.5, 0.5]])


def test_dask_chunked_over_time(make_fake_raster):
    """Chunking over a non-core dim must compute and match the eager result."""
    pytest.importorskip("dask")

    layers = [make_fake_raster(3, 20, 20, seed=seed) for seed in (1, 2)]

    # generate data
    cube = xr.DataArray(np.stack(layers), dims=("time", "class", "y", "x"))

    # smooth chunk
    result = smoother.smooth(cube.chunk({"time": 1})).compute()

    # smooth
    eager = smoother.smooth(cube)

    np.testing.assert_array_equal(result.values, eager.values)


def test_dask_chunked_spatial_raises(make_dataarray):
    """Chunking over a spatial (core) dim must raise an error."""
    pytest.importorskip("dask")

    da = make_dataarray().chunk({"y": 10})

    with pytest.raises(ValueError):
        smoother.smooth(da)
