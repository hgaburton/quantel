#!/bin/bash
set -euo pipefail

# Build the C++ extension, installing it directly into site-packages
rm -rf build
mkdir -p build && cd build
cmake "$SRC_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$PYTHON" \
    -DQUANTEL_INSTALL_DIR="$SP_DIR/quantel/lib"
make -j"${CPU_COUNT:-4}" install
cd "$SRC_DIR"

# Install the Python package source files into site-packages
mkdir -p "$SP_DIR/quantel"
cp quantel/__init__.py quantel/main.py "$SP_DIR/quantel/"
cp -r quantel/drivers quantel/gnme quantel/ints quantel/io quantel/opt quantel/utils quantel/wfn "$SP_DIR/quantel/"

# pygnme is not on conda-forge; install it separately after:
#   pip install git+https://github.com/hgaburton/pygnme.git
