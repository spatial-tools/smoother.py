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

"""Generic entry point for the smoothing methods."""

from smoother.bayes import bayes

METHODS = {
    "bayes": bayes,
}


def smooth(data, method="bayes", **kwargs):
    """Smooth a probability raster from an image classification.

    This is the single entry point of the package. It selects a smoothing
    ``method`` and forwards the remaining options to the backend that matches
    the type of ``data`` (``numpy.ndarray`` or ``xarray.DataArray``).

    Args:
        data: Probability raster from a classification. Either a
              ``numpy.ndarray`` with shape ``(class, y, x)`` or an
              ``xarray.DataArray`` with class, y and x dimensions. DataArrays
              with extra dimensions (e.g. ``time``) are smoothed once per layer,
              and dask-backed DataArrays are processed in parallel.

        method (str): Smoothing method to apply. Available: ``"bayes"``
                      (Bayesian smoothing with spatial predictors).

        window_size (int): Size of the smoothing neighbourhood
                           (min = 3, max = 21). Default: ``7``.

        smoothness (int): Estimated variance of the logit of class
                          probabilities (min = 1, max = 200). A larger value
                          produces a stronger smoothing effect. Default: ``10``.

        neigh_fraction (float): Fraction of neighbours with the highest
                                probabilities used in the Bayesian inference.
                                Default: ``0.5``.

        class_dim (str | None): Name of the class dimension. ``xarray`` input
                                only. Inferred by name when omitted.

        y_dim (str | None): Name of the y dimension. ``xarray`` input only.
                            Inferred by name when omitted.

        x_dim (str | None): Name of the x dimension. ``xarray`` input only.
                            Inferred by name when omitted.

    Returns:
        The smoothed probability raster, in the same type and shape as ``data``.
        For ``xarray.DataArray`` input, dimensions, coordinates, attributes and
        the CRS are preserved.

    Note:
        ``class_dim``, ``y_dim`` and ``x_dim`` only apply to ``xarray.DataArray``
        input. Passing them with a ``numpy.ndarray`` raises a ``TypeError``.

        To learn more about the spatial smoothing, see the ``sits`` book:
        https://e-sensing.github.io/sitsbook/cl_smoothing.html

    Info:
        The smoothing was adapted from the ``sits`` R package. Thanks to
        Gilberto Camara, Rolf Simoes and Felipe Carvalho, the authors of the
        original smooth function in R.
    """
    try:
        method_fn = METHODS[method]

    except KeyError:
        raise ValueError(
            f"unknown method {method!r}; available methods are {sorted(METHODS)}"
        ) from None

    return method_fn(data, **kwargs)
