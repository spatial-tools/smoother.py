# Pyodide build

Script to build a smoother wheel for [Pyodide](https://pyodide.org), Python compiled to WebAssembly.

## How to build

The `Pyodide` build tools must run on the Python version the `Pyodide` release targets. For `Pyodide` 314.x that is Python 3.14.

```sh
uv tool install --python 3.14 pyodide-cli --with pyodide-build

scripts/wasm/build.sh
```

The script installs the `Pyodide` cross-build environment and its Emscripten, and writes the wheel to `dist/wasm`. Set `PYODIDE_VERSION` to target another release.

## How to use

The wheel is published on PyPI, so `micropip` installs it directly:

```js
const pyodide = await loadPyodide();
await pyodide.loadPackage("micropip");
await pyodide.runPythonAsync(`
import micropip
await micropip.install("smoother-py")

import numpy as np
import smoother

probs = np.random.default_rng(42).dirichlet(np.ones(4), size=(100, 100)).transpose(2, 0, 1)
smoothed = smoother.smooth(probs, method="bayes")
`);
```

## Limits

- It runs about 3x slower than the native build;
- Results may differ from the native build by floating-point rounding (around `1e-16`);
- The wheel is built for one Python version, not as `abi3`, because nanobind does not build in stable ABI mode under `Pyodide`.
