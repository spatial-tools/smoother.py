#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

using DoubleMatrix =
    nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

using DoubleVector =
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// compute outside indices of a vector as a mirror
std::vector<int64_t> locus_neigh(int size, int leg) {
    std::vector<int64_t> res(size + 2 * leg);

    for (int i = 0; i < static_cast<int>(res.size()); ++i) {
        if (i < leg) {
            res[i] = leg - i - 1;
        }
        
        else if (i < size + leg) {
            res[i] = i - leg;
        }
        
        else {
            res[i] = 2 * size + leg - i - 1;
        }
    }

    return res;
}

// Code adapted from the sits R package.
// You can learn more using the following link:
//  https://github.com/e-sensing/sits/blob/master/src/smooth.cpp
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

    // initialize result vectors
    double *res = new double[npixels * nbands];
    std::vector<double> neigh(window_size * window_size);

    // compute window leg
    int leg = window_size / 2;

    // compute locus mirror
    std::vector<int64_t> loci = locus_neigh(nrows, leg);
    std::vector<int64_t> locj = locus_neigh(ncols, leg);

    // compute number of neighbors to be used
    int neigh_high = std::ceil(neigh_fraction * window_size * window_size);

    // compute values for each pixel
    for (int i = 0; i < nrows; ++i)
    {
        for (int j = 0; j < ncols; ++j)
        {
            // for all bands
            for (size_t band = 0; band < nbands; ++band)
            {
                // compute the neighborhood
                for (int wi = 0; wi < window_size; ++wi)
                {
                    for (int wj = 0; wj < window_size; ++wj)
                    {
                        neigh[wi * window_size + wj] =
                            logits(loci[wi + i] * ncols + locj[wj + j], band);
                    }
                }
                if (neigh_fraction < 1.0)
                {
                    // Move the highest neigh_high logit values to the front
                    // (in descending order)
                    std::partial_sort(neigh.begin(),
                                      neigh.begin() + neigh_high,
                                      neigh.end(),
                                      std::greater<double>());
                }
                
                // get the estimates for prior
                // normal with mean m0 and variance s0
                // (sample variance, N - 1 denominator)
                double sum = 0.0;
                
                for (int k = 0; k < neigh_high; ++k) {
                    sum += neigh[k];
                }
                    
                double m0 = sum / neigh_high;
                double sq_sum = 0.0;

                for (int k = 0; k < neigh_high; ++k) {
                    double dev = neigh[k] - m0;
                    sq_sum += dev * dev;
                }

                double s0 = neigh_high > 1 ? sq_sum / (neigh_high - 1) : 0.0;
                
                // get the current value
                double x0 = logits(i * ncols + j, band);
                
                // weight for Bayesian estimator
                double w = s0 / (s0 + smoothness(band));
                
                // apply Bayesian smoother
                res[(i * ncols + j) * nbands + band] = w * x0 + (1 - w) * m0;
            }
        }
    }

    // create a capsule to own the result array
    nb::capsule owner(res, [](void *p) noexcept
                      { delete[] static_cast<double *>(p); });

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
