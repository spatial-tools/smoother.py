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
from smoother.bayes import bayes


@bayes.register(np.ndarray)
def bayes_numpy(
    data: np.ndarray,
    window_size: int = 7,
    smoothness: int = 10,
    neigh_fraction: float = 0.5,
) -> np.ndarray:
    """Bayesian smoothing of a ``(class, y, x)`` array."""
    nclasses, nrows, ncols = data.shape

    # prepare smoothness
    smoothness = np.repeat(smoothness, nclasses).astype(np.float64)

    # compute logits
    logit = np.log(data / (np.sum(data, axis=0, keepdims=True) - data))
    logit = np.ascontiguousarray(logit.reshape(nclasses, -1).T)

    # smooth
    data = _smoother.bayes(logit, nrows, ncols, window_size, smoothness, neigh_fraction)

    # inverse logit
    data = np.exp(data) / (np.exp(data) + 1)
    data = data.T

    # return data in the same input shape
    return data.reshape((nclasses, nrows, ncols))
