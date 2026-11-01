#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <https://www.gnu.org/licenses/>.
#

"""Behaviour shared with sits's ``sits_smooth()`` (sits 1.5.4)."""

import numpy as np
import pytest

import smoother


def test_eight_bit_cube_has_no_nan():
    """Quantized input (many exact 0s and 1s) must not turn into NaN."""
    rng = np.random.default_rng(1)

    # 8 classes, 40x40 pixels
    probs = rng.dirichlet(np.full(8, 0.03), size=(40, 40)).transpose(2, 0, 1)

    # round to 8-bit
    data = np.round(probs * 255) / 255

    # smooth
    result = smoother.smooth(data)

    # check that no NaN values were produced
    assert not np.any(np.isnan(result))

    # check that all values are in the [0, 1] range
    assert np.all((result >= 0) & (result <= 1))


def test_exact_zero_and_one_are_clamped_like_sits():
    """sits sets exact 1 to 0.999999 and exact 0 to 0.000001."""
    # create a 2x5x5 array with exact 0s and 1s
    data = np.zeros((2, 5, 5))
    data[0] = 1.0

    # smooth
    result = smoother.smooth(data, window_size=3)

    # check that the exact 0s and 1s were clamped
    np.testing.assert_allclose(result[0], 0.999999)
    np.testing.assert_allclose(result[1], 0.000001)


def test_nan_pixel_takes_neighbourhood_value():
    """NaN neighbours are ignored and a NaN pixel gets the neighbourhood mean."""
    # create a 2x5x5 array with 0.5s, except for a NaN in the center
    data = np.full((2, 5, 5), 0.5)
    data[0] = 0.8
    data[1] = 0.2
    data[:, 2, 2] = np.nan

    # smooth
    result = smoother.smooth(data, window_size=3)

    # check that the NaN pixel got the neighbourhood mean
    np.testing.assert_allclose(result[:, 2, 2], [0.8, 0.2])
    assert not np.any(np.isnan(result))


def test_per_class_smoothness():
    """Smoothness is applied per class: a larger value smooths more."""
    # create a 30x30x2 array with random probabilities
    rng = np.random.default_rng(3)
    data = rng.dirichlet(np.ones(2), size=(30, 30)).transpose(2, 0, 1)

    # smooth
    result = smoother.smooth(data, smoothness=[1, 200])

    # check that the second class changed more
    change = np.abs(result - data).mean(axis=(1, 2))

    # check that the second class changed more than the first
    assert change[1] > 2 * change[0]


def test_defaults_are_sits_defaults(make_fake_raster):
    """Defaults: window_size=9, smoothness=20, neigh_fraction=0.5."""
    data = make_fake_raster(nclasses=3, nrows=12, ncols=12)

    # smooth
    np.testing.assert_array_equal(
        smoother.smooth(data),
        smoother.smooth(data, window_size=9, smoothness=20, neigh_fraction=0.5),
    )


def test_window_larger_than_image(make_fake_raster):
    """A window larger than the image works (edges are mirrored)."""
    data = make_fake_raster(nclasses=3, nrows=2, ncols=3)

    # smooth
    result = smoother.smooth(data, window_size=21)

    # check that no NaN values were produced
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 4},
        {"window_size": 1},
        {"neigh_fraction": 0},
        {"neigh_fraction": 1.5},
        {"smoothness": 0},
        {"smoothness": [10, 20]},
    ],
    ids=repr,
)
def test_invalid_parameters_raise(kwargs, make_fake_raster):
    """Bad parameters raise ValueError."""
    data = make_fake_raster(nclasses=3, nrows=10, ncols=10)

    # smooth
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        smoother.smooth(data, **kwargs)


@pytest.mark.parametrize("shape", [(10, 10), (3, 0, 5)])
def test_invalid_data_shape_raises(shape):
    """Data must be a non-empty (class, y, x) array."""
    with pytest.raises(ValueError, match="data"):
        smoother.smooth(np.full(shape, 0.5))


def test_empty_dataarray_raises():
    """An empty xarray dimension raises instead of crashing the native code."""
    xr = pytest.importorskip("xarray")
    da = xr.DataArray(np.full((3, 0, 5), 0.5), dims=("class", "y", "x"))

    with pytest.raises(ValueError, match="data"):
        smoother.smooth(da)


def test_input_is_not_mutated():
    """The caller's array is unchanged, including exact 0s and 1s."""
    data = np.zeros((2, 5, 5))
    data[0] = 1.0
    before = data.copy()

    smoother.smooth(data)

    np.testing.assert_array_equal(data, before)
