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

"""Numpy backend for the smoother."""

import numpy as np

from smoother import _smoother
from smoother.arrays import class_smoothness
from smoother.bayes import (
    NEIGH_FRACTION,
    SMOOTHNESS,
    WINDOW_SIZE,
    bayes,
)


@bayes.register(np.ndarray)
def bayes_numpy(
    data: np.ndarray,
    window_size: int = WINDOW_SIZE,
    smoothness: float | list[float] = SMOOTHNESS,
    neigh_fraction: float = NEIGH_FRACTION,
) -> np.ndarray:
    """Bayesian smoothing of a ``(class, y, x)`` array."""
    nclasses, nrows, ncols = data.shape

    # one smoothness value per class
    smoothness = class_smoothness(smoothness, nclasses)

    # work on a float64 copy, so the caller's array is never changed
    values = np.array(data, dtype=np.float64)

    # avoid -Inf or +Inf in logits, as sits does: only exact 0 and 1 change
    values[values == 1.0] = 0.999999
    values[values == 0.0] = 0.000001

    # compute logits
    logit = np.log(values / (np.sum(values, axis=0, keepdims=True) - values))
    logit = np.ascontiguousarray(logit.reshape(nclasses, -1).T)

    # smooth
    logit = _smoother.bayes(
        logit, nrows, ncols, window_size, smoothness, neigh_fraction
    )

    # inverse logit
    values = np.exp(logit) / (np.exp(logit) + 1)

    # return data in the same input shape
    return values.T.reshape((nclasses, nrows, ncols))
