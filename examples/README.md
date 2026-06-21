# Examples

Usage example of `smoother.py` on a real classification output.

## Data

[`data/`](./data) holds a [`sits`](https://github.com/e-sensing/sits) classification probs GeoTIFF with one band per land-use class, in this order:

| Band | Class    | Color     |
|------|----------|-----------|
| 1    | Cerrado  | `#a2d43f` |
| 2    | Forest   | `#1E8449` |
| 3    | Pasture  | `#FAD12D` |
| 4    | Soy_Corn | `#D68910` |

The colors come from the QGIS style `data/style.qml` and are reused in the notebook for plotting.

## Running

Install the example dependencies:

```bash
pip install "smoother-py[examples]"
# or, in this repo:  uv sync --extra examples
```

Then open the notebook:

```bash
jupyter lab bayes.ipynb
```
