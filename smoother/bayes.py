#
# Copyright (C) 2024-2026 smoother.py.
#
# smoother.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
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
