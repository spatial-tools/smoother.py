## smoother.py 🧹

Spatial smoothing methods for probability maps.

### Installation

To install the package, you can use pip:

```bash
pip install smoother-py
```

### Methods available

Currently, the `smoother` implements the [Bayes smoothing method](https://e-sensing.github.io/sitsbook/cl_smoothing.html), adapting the code from the [sits](https://github.com/e-sensing/sits) R package.

### Usage

First, import the package:

```python
import smoother
```

Then, use the `smooth` function on the `numpy` data of your map. The smoothing method is selected with `method` (defaults to `"bayes"`), and any method-specific options are passed as keyword arguments:

```python
smoothed = smoother.smooth(your_numpy_array)

# selecting the method explicitly and tuning it
smoothed = smoother.smooth(your_numpy_array, method="bayes", window_size=5)
```

The `bayes` method gives the same result as `sits_smooth()` from sits 1.5.4. Its defaults are the same as sits: `window_size=9`, `smoothness=20` and `neigh_fraction=0.5`. Exact `0` and `1` probabilities are handled as sits does, so quantized input (e.g. 8-bit) works. Invalid parameters raise `ValueError`.

You can use [rasterio](https://rasterio.readthedocs.io/en/latest/index.html) or other library to load the numpy array.

#### xarray

`smooth` also accepts `xarray` objects. To use it, first install the `xarray` dependencies:

```bash
pip install "smoother-py[xarray]"
```

Then, using your `xarray` data, you can call the `smooth` function:

```python
# DataArray with class, y and x dimensions
smoothed = smoother.smooth(your_data_array)

# dimension names are inferred (e.g. band/class, y/latitude, x/longitude)
# you can also name them explicitly
smoothed = smoother.smooth(your_data_array, class_dim="band")
```

Extra dimensions (e.g. `time`) are smoothed per layer automatically.

#### Discovering the options

`smoother.smooth` documents every option, the smoothing parameters and the xarray-only dimension names:

```python
help(smoother.smooth)
```

### Development

The project uses [uv](https://docs.astral.sh/uv/) to manage the development environment:

```bash
# create the environment and build/install the package
uv sync

# run the tests
uv run pytest
```

### Learn more

The smoothing methods implemented in this package uses the approach described in the [sits documentation](https://e-sensing.github.io/sitsbook/bayesian-smoothing-for-post-processing.html). This method helps in refining the probability maps by considering the spatial context of each pixel.

### Acknowledgments

We would like to thank the developers and contributors of the `sits` R package for their work on spatial smoothing methods for Earth observation data. Their comprehensive [documentation](https://e-sensing.github.io/sitsbook/) and methodologies have been invaluable in the development of `smoother.py`.

### Contributing

We welcome contributions! If you have suggestions for improvements or bug fixes, please feel free to fork the repository and submit a pull request.

### License

Code is licensed under the **GNU General Public License v2.0**. See the [LICENSE](LICENSE) file. The bayes smooth code is ported from the [`sits`](https://github.com/e-sensing/sits) R package.
