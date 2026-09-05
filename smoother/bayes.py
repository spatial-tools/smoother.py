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


@singledispatch
def bayes(
    data,
    window_size: int = 7,
    smoothness: int = 10,
    neigh_fraction: float = 0.5,
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
