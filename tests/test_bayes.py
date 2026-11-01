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

"""Test Bayesian smoothing with synthetic rasters."""

import numpy as np
import pytest

import smoother


def test_output_shape(make_fake_raster):
    """The output shape must match the input shape."""
    data = make_fake_raster(nclasses=4, nrows=50, ncols=50)

    result = smoother.smooth(data)

    assert result.shape == data.shape


def test_output_range(make_fake_raster):
    """All values must be valid probabilities in [0, 1]."""
    data = make_fake_raster(nclasses=5, nrows=40, ncols=40)

    result = smoother.smooth(data)

    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_uniform_input_stays_uniform():
    """Uniform probabilities should remain roughly uniform after smoothing."""
    nclasses = 3
    data = np.full((nclasses, 20, 20), 1.0 / nclasses)

    result = smoother.smooth(data)

    np.testing.assert_allclose(result, 1.0 / nclasses, atol=1e-5)


def test_custom_parameters(make_fake_raster):
    """Verify smoothing runs with non-default parameters."""
    data = make_fake_raster(nclasses=3, nrows=25, ncols=25)

    result = smoother.smooth(data, window_size=5, smoothness=20, neigh_fraction=0.3)

    assert result.shape == data.shape
    assert np.all(np.isfinite(result))


def test_unknown_method_raises(make_fake_raster):
    """Selecting a method that does not exist must raise an error."""
    data = make_fake_raster(nclasses=3, nrows=10, ncols=10)

    with pytest.raises(ValueError, match="unknown method"):
        smoother.smooth(data, method="does-not-exist")
