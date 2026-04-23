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
CMake will automatically download or extract `fmt` (11.1.2), `libint2` (2.9.0), and
`armadillo` (12.8.3) into `external/` if they are not already present.
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
