#
# Copyright (C) 2024-2026 smoother.py.
#
# smoother.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""x-array backend for the smoother."""

import numpy as np
import xarray as xr

from smoother.backends.numpy import bayes_numpy
from smoother.bayes import bayes

# Possible dimension names
CLASS_DIM_NAMES = ("class", "classes", "band", "bands", "variable")

# Possible dimension names for the y dimension
Y_DIM_NAMES = ("y", "lat", "latitude")

# Possible dimension names for the x dimension
X_DIM_NAMES = ("x", "lon", "long", "longitude")


def infer_dims(data, class_dim, y_dim, x_dim):
    """Resolve the class, y and x dimension names of a DataArray.

    Resolution order per dimension:
        o-> explicit name (validated) ->
        o-> inference from common names ->
        o-> positional ``(class, y, x)`` fallback (for 3D input only).
    """
    dim_3d_size = 3

    requested = (
        ("class_dim", class_dim, CLASS_DIM_NAMES),
        ("y_dim", y_dim, Y_DIM_NAMES),
        ("x_dim", x_dim, X_DIM_NAMES),
    )

    # explicit names are always prioritized
    resolved = {}

    for param, given, _ in requested:
        # validate the explicit name
        if given is not None and given not in data.dims:
            # raise an error if the explicit name is not a dimension
            raise ValueError(
                f"{param}={given!r} is not a dimension of the input; "
                f"dimensions are {tuple(data.dims)}"
            )

        # otherwise, use the explicit name
        resolved[param] = given

    # for the missing dimensions, infer from common names
    for param, _, candidates in requested:
        # check if the dimension is not defined yet, test common name
        if resolved[param] is None:
            # check the common names for a match
            resolved[param] = next(
                (
                    dim
                    for dim in data.dims
                    if str(dim).lower() in candidates and dim not in resolved.values()
                ),
                None,
            )

    # on 3D input, the dimensions left over fill the open slots in
    # (class, y, x) order
    if None in resolved.values() and data.ndim == dim_3d_size:
        # find the leftover dimensions
        leftover = [dim for dim in data.dims if dim not in resolved.values()]

        # fill identified dimensions
        for param, _, _ in requested:
            # check if is empty
            if resolved[param] is None:
                resolved[param] = leftover.pop(0)

    # check for missing dimensions
    if None in resolved.values() or len(set(resolved.values())) != dim_3d_size:
        # raise an issue in such case
        raise ValueError(
            "could not identify the `class`, `y` and `x` dimensions of the input "
            f"(dimensions are {tuple(data.dims)}); pass `class_dim`, `y_dim` "
            "and `x_dim` explicitly"
        )

    # return the resolved names
    return resolved["class_dim"], resolved["y_dim"], resolved["x_dim"]


@bayes.register(xr.DataArray)
def bayes_dataarray(
    data: xr.DataArray,
    window_size: int = 7,
    smoothness: int = 10,
    neigh_fraction: float = 0.5,
    class_dim: str | None = None,
    y_dim: str | None = None,
    x_dim: str | None = None,
) -> xr.DataArray:
    """Bayesian smoothing of a DataArray with class, y and x dims."""
    # infer the dimension names
    class_dim, y_dim, x_dim = infer_dims(data, class_dim, y_dim, x_dim)

    # apply the smoothing
    result = xr.apply_ufunc(
        bayes_numpy,
        data,
        input_core_dims=[[class_dim, y_dim, x_dim]],
        output_core_dims=[[class_dim, y_dim, x_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        keep_attrs=True,
        kwargs={
            "window_size": window_size,
            "smoothness": smoothness,
            "neigh_fraction": neigh_fraction,
        },
    )

    # `apply_ufunc` moves core dims last, so we need to
    # restore the original order
    return result.transpose(*data.dims)
