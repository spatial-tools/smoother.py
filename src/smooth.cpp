#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

using DoubleMatrix =
    nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

using DoubleVector =
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// compute outside indices of a vector as a mirror
// (same as sits). Indices that still fall outside the vector, when the
// window is larger than the image, are mirrored again until they fit
// (numpy "symmetric" padding).
std::vector<int64_t> locus_neigh(int size, int leg) {
    std::vector<int64_t> res(size + 2 * leg);
    const int64_t period = 2 * static_cast<int64_t>(size);

    for (int i = 0; i < static_cast<int>(res.size()); ++i) {
        int64_t index;

        if (i < leg) {
            index = leg - i - 1;
        }

        else if (i < size + leg) {
            index = i - leg;
        }

        else {
            index = 2 * static_cast<int64_t>(size) + leg - i - 1;
        }

        index %= period;
        if (index < 0) {
            index += period;
        }

        res[i] = index < size ? index : period - 1 - index;
    }

    return res;
}

// Code adapted from the sits R package (1.5.4).
// You can learn more using the following link:
//  https://github.com/e-sensing/sits/blob/master/src/smooth_bayes.cpp
nb::ndarray<nb::numpy, double, nb::ndim<2>> bayes_smooth(
    DoubleMatrix logits,
    const int nrows,
    const int ncols,
    const int window_size,
    DoubleVector smoothness,
    const double neigh_fraction)
{
    const size_t npixels = logits.shape(0);
    const size_t nbands = logits.shape(1);

    if (nrows < 0 || ncols < 0 ||
        npixels != static_cast<size_t>(nrows) * static_cast<size_t>(ncols)) {
        throw std::invalid_argument("logits rows must equal nrows * ncols");
    }

    if (window_size < 1 || window_size % 2 == 0) {
        throw std::invalid_argument("window_size must be a positive odd integer");
    }

    if (smoothness.shape(0) != nbands) {
        throw std::invalid_argument("smoothness must have one value per class");
    }

    // initialize result vectors
    double *res = new double[npixels * nbands];

    // create a capsule to own the result array
    nb::capsule owner(res, [](void *p) noexcept
                      { delete[] static_cast<double *>(p); });

    // an empty image has nothing to smooth (and no edge to mirror)
    if (npixels == 0) {
        return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
            res, {npixels, nbands}, owner);
    }

    std::vector<double> neigh(
        static_cast<size_t>(window_size) * static_cast<size_t>(window_size));

    // compute window leg
    int leg = window_size / 2;

    // compute locus mirror
    std::vector<int64_t> loci = locus_neigh(nrows, leg);
    std::vector<int64_t> locj = locus_neigh(ncols, leg);

    // compute values for each pixel
    for (int i = 0; i < nrows; ++i)
    {
        for (int j = 0; j < ncols; ++j)
        {
            // for all bands
            for (size_t band = 0; band < nbands; ++band)
            {
                // compute the neighborhood, dropping NaN values
                size_t nvalid = 0;

                for (int wi = 0; wi < window_size; ++wi)
                {
                    for (int wj = 0; wj < window_size; ++wj)
                    {
                        double value =
                            logits(loci[wi + i] * ncols + locj[wj + j], band);

                        if (!std::isnan(value)) {
                            neigh[nvalid++] = value;
                        }
                    }
                }

                // compute number of neighbors to be used
                size_t neigh_high = static_cast<size_t>(
                    std::ceil(neigh_fraction * static_cast<double>(nvalid)));

                if (neigh_fraction < 1.0)
                {
                    // Move the highest neigh_high logit values to the front
                    // (in descending order)
                    std::partial_sort(neigh.begin(),
                                      neigh.begin() + neigh_high,
                                      neigh.begin() + nvalid,
                                      std::greater<double>());
                }

                // get the estimates for prior
                // normal with mean m0 and variance s0
                // (sample variance, N - 1 denominator)
                double m0 = std::numeric_limits<double>::quiet_NaN();
                double s0 = std::numeric_limits<double>::quiet_NaN();

                if (neigh_high > 0) {
                    double sum = 0.0;

                    for (size_t k = 0; k < neigh_high; ++k) {
                        sum += neigh[k];
                    }

                    m0 = sum / static_cast<double>(neigh_high);

                    double sq_sum = 0.0;

                    for (size_t k = 0; k < neigh_high; ++k) {
                        double dev = neigh[k] - m0;
                        sq_sum += dev * dev;
                    }

                    // one value has no spread (sits returns NaN here)
                    s0 = neigh_high > 1 ? sq_sum / static_cast<double>(neigh_high - 1) : 0.0;
                }

                // get the current value
                double x0 = logits(i * ncols + j, band);
                double *out = &res[(i * ncols + j) * nbands + band];

                if (std::isnan(x0) || s0 < 1e-04) {
                    *out = m0;
                } else {
                    // weight for Bayesian estimator
                    double w = s0 / (s0 + smoothness(band));

                    // apply Bayesian smoother
                    *out = w * x0 + (1 - w) * m0;
                }
            }
        }
    }

    // prepare the result as a numpy array
    return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
        res, {npixels, nbands}, owner);
}

NB_MODULE(_smoother, m) {
    // Optional docstring
    m.doc() = "(Internal) Spatial Smoothing Functions";

    // define the bayes smoothing function
    m.def("bayes", &bayes_smooth, "Bayes Smooth");
}
