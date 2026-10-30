#!/usr/bin/env bash
set -eu

PYODIDE_VERSION="${PYODIDE_VERSION:-314.0.7}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/dist/wasm"

command -v pyodide >/dev/null 2>&1 || { echo "error: pyodide not found" >&2; exit 1; }

# install Pyodide cross-build environment and Emscripten
pyodide xbuildenv install "$PYODIDE_VERSION"
pyodide xbuildenv use "$PYODIDE_VERSION"
pyodide xbuildenv install-emscripten

# nanobind does not build in stable ABI (abi3) mode under Pyodide, so it is disabled here
rm -f "$OUT"/smoother_py-*.whl
(cd "$ROOT" && pyodide build -C wheel.py-api= -o "$OUT")

echo "wheel written to $(ls "$OUT"/smoother_py-*.whl)"
