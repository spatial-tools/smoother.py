#
# Copyright (C) 2024 smoother.py.
#
# smoother.py is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

import numpy as np

from smoother import _smoother


def bayes(
    data: np.ndarray,
    window_size: int = 7,
    smoothness: int = 10,
    neigh_fraction: float = 0.5,
) -> np.ndarray:
    """Smooth probability raster with spatial predictors.

    Args:
        data (np.array): Probability raster from a classification with the
                         following shape (Number of classes, X, Y).

        window_size (int): Size of the smooth neighbourhood
                           (min = 3, max = 21).

        smoothness (int): Estimated variance of logit of class probabilities
                          (min = 1, max = 200).

        neigh_fraction (float): Fraction of neighbors with high probabilities
                                to be used in Bayesian inference.

    Note:
        To learn more about the spatial smooth, you can check the following
        book resource:
        https://e-sensing.github.io/sitsbook/bayesian-smoothing-for-post-processing.html

    Info:
        This function was adapted from the `sits` R package. Thanks to
        Gilberto Camara and Rolf Simoes, the authors of the original smooth
        function in R.
    """
    nclasses, nrows, ncols = data.shape

    # prepare smoothness
    smoothness = np.repeat(smoothness, nclasses)

    # compute logits
    logit = np.log(data / (np.sum(data, axis=0, keepdims=True) - data))
    logit = logit.reshape(nclasses, -1).T

    # smooth
    data = _smoother.bayes(logit, nrows, ncols, window_size, smoothness, neigh_fraction)

    # inverse logit
    data = np.exp(data) / (np.exp(data) + 1)
    data = data.T

    # return data in the same input shape
    return data.reshape((nclasses, nrows, ncols))
