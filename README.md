# QuantEl

A library for performing state-specific electronic structure and testing new methods.

Any questions should be directed to Hugh Burton (hgaburton[@]gmail.com)
   
## Installation Instructions <a name="install">

### Prerequisites

QuantEl requires a number of external libraries. 
Most of these can be easily installed using a conda environment. 
The C++ dependencies (`fmt`, `libint2`, and `armadillo`) are fetched and set up automatically by CMake.

### Option A — conda package (recommended)

Build and install quantel as a conda package into your base or target environment:
```
conda build conda-recipe/
conda install --use-local quantel
```
This handles all C++ and Python dependencies automatically.

### Option B — development install

#### 1. Create the conda environment
```
conda env create -f environment.yml
conda activate quantel
```
This installs all required build tools (`cmake`, `pybind11`, `boost`, `eigen`, `llvm-openmp`)
and Python packages (`numpy`, `scipy`, `h5py`, `pandas`, `pyscf`, etc.).

#### 2. PyGNME
The nonorthogonal matrix elements in QuantEl require `pygnme`. Install it after the
conda environment is active (the `--no-build-isolation` flag is required so the build
can find the conda-installed numpy headers):
```
pip install --no-build-isolation git+https://github.com/hgaburton/pygnme.git
```

#### 3. Build the C++ extension
All C++ dependencies (`fmt`, `libint2`, `armadillo`) are provided by the conda environment.
```
mkdir build && cd build
cmake ../
make -j4 install
cd ../
```

#### 4. Add to your Python path
```
export PYTHONPATH=~/PATH/TO/QUANTEL:$PYTHONPATH
```

### Linux note — OpenBLAS and OpenMP

On Linux, OpenBLAS (the default conda BLAS) is compiled with pthreads, which conflicts
with quantel's OpenMP parallelism and produces a runtime warning. The `environment.yml`
sets `OPENBLAS_NUM_THREADS=1` automatically when the environment is created. If you
already have the environment, apply it manually:
```
conda env config vars set OPENBLAS_NUM_THREADS=1 -n quantel
conda activate quantel
```
