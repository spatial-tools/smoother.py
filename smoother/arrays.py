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

"""Array helpers shared by the smoothing methods and backends."""

import numpy as np

# Number of dimensions of a (class, y, x) array
DATA_NDIM = 3


def check_smoothness(smoothness, nclasses):
    """Check that ``smoothness`` is valid for ``nclasses`` classes.

    ``smoothness`` must be a number, or a sequence with one number per class.
    All values must be greater than 0.

    Raises:
        `ValueError` when ``smoothness`` is not valid.
    """
    try:
        values = np.asarray(smoothness)

    except ValueError:
        raise ValueError(
            f"smoothness must be a number or a sequence of numbers, got {smoothness!r}"
        ) from None

    if values.dtype.kind not in "iuf" or values.ndim > 1:
        raise ValueError(
            f"smoothness must be a number or a sequence of numbers, got {smoothness!r}"
        )

    if values.ndim == 1 and values.shape[0] != nclasses:
        raise ValueError(
            f"smoothness has {values.shape[0]} values, but the data has "
            f"{nclasses} classes; pass one value per class or a single number"
        )

    if not np.all(values > 0):
        raise ValueError(f"smoothness must be greater than 0, got {smoothness!r}")


def class_smoothness(smoothness, nclasses):
    """Return one contiguous ``float64`` smoothness value per class.

    A number is repeated for every class.

    Raises:
        `ValueError` when ``smoothness`` is not valid (see ``check_smoothness``).
    """
    check_smoothness(smoothness, nclasses)

    values = np.asarray(smoothness, dtype=np.float64)

    return np.ascontiguousarray(np.broadcast_to(values, (nclasses,)))


def check_data_shape(shape):
    """Check that ``shape`` is a non-empty ``(class, y, x)`` shape.

    Raises:
        `ValueError` when the shape is not valid.
    """
    if len(shape) != DATA_NDIM or 0 in shape:
        raise ValueError(
            "data must be a non-empty 3-D array with shape (class, y, x), "
            f"got shape {tuple(shape)}"
        )
