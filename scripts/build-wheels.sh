#!/bin/bash

set -e

echo "Building wheels"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info

find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo "Building wheels..."
uv build
