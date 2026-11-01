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

"""Bayesian smoothing of probability rasters."""

from functools import singledispatch
from numbers import Integral, Real

import numpy as np

from smoother.arrays import check_data_shape, check_smoothness

# Default parameters, the same as sits ``sits_smooth()``
WINDOW_SIZE = 9
SMOOTHNESS = 20
NEIGH_FRACTION = 0.5

# Smallest window allowed (sits documents 3, and checks 5 internally)
MIN_WINDOW_SIZE = 3


@singledispatch
def bayes(
    data,
    window_size: int = WINDOW_SIZE,
    smoothness: float = SMOOTHNESS,
    neigh_fraction: float = NEIGH_FRACTION,
    **kwargs,
):
    """Bayesian smoothing of a probability raster (internal dispatch target).

    This is the ``functools.singledispatch`` generic that selects the backend
    for the type of ``data`` (``numpy.ndarray`` or ``xarray.DataArray``). It is
    not part of the public API.

    Raises:
        `TypeError` when data input is not supported.
    """

    raise TypeError(
        f"bayes() does not support {type(data).__name__}; expected "
        "numpy.ndarray or xarray.DataArray (xarray support requires "
        '`pip install "smoother-py[xarray]"`)'
    )


def validate_bayes(
    data,
    window_size=WINDOW_SIZE,
    smoothness=SMOOTHNESS,
    neigh_fraction=NEIGH_FRACTION,
    **_,
):
    """Check the parameters of the ``bayes`` method, as sits does.

    Raises:
        `ValueError` when a parameter is not valid.
    """
    if (
        not isinstance(window_size, Integral)
        or isinstance(window_size, bool)
        or window_size < MIN_WINDOW_SIZE
        or window_size % 2 == 0
    ):
        raise ValueError(
            f"window_size must be an odd integer >= {MIN_WINDOW_SIZE}, "
            f"got {window_size!r}"
        )

    if (
        not isinstance(neigh_fraction, Real)
        or isinstance(neigh_fraction, bool)
        or not 0 < neigh_fraction <= 1
    ):
        raise ValueError(f"neigh_fraction must be in (0, 1], got {neigh_fraction!r}")

    # the class count is only known here for numpy input. The xarray backend
    # checks the shape and per-class length once the class dimension is known
    if isinstance(data, np.ndarray):
        # check the shape
        check_data_shape(data.shape)

        # check the smoothness
        check_smoothness(smoothness, data.shape[0])

    else:
        # check the smoothness
        check_smoothness(smoothness, np.size(smoothness))
