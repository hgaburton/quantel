# QuantEl

A library for performing state-specific electronic structure and testing new methods.

Any questions should be directed to Hugh Burton (hgaburton[@]gmail.com)
   
## Installation Instructions <a name="install">

### Prerequisites

QuantEl requires a number of external libraries. 
Most of these can be easily installed using a conda environment, but `fmt`, `libint` and `armadillo` currently need to be installed manually. 
These manual external libraries must be setup in the directory `./external/` following the instructions below.

#### Conda environment
The conda environment `quantel` can be setup by running
```
conda env create -f environment.yml
conda activate quantel
```

#### fmt 
Library to support easier C++ string formatting. From the directory `external/`, run
```
git clone https://github.com/fmtlib/fmt.git
```

#### libint2
Integral library used by QuantEl. From the directory `external/`, run
```
wget https://github.com/evaleev/libint/releases/download/v2.9.0/libint-2.9.0.tgz
tar -xvf libint-2.9.0.tgz
mv libint-2.9.0 libint2
```
#### armadillo
C++ linear algebra library used by QuantEl. From the directory `external/`, run
```
wget https://sourceforge.net/projects/arma/files/armadillo-12.8.3.tar.xz
tar -xvf armadillo-12.8.3.tar.xz
mv armadillo-12.8.3 armadillo
```

### Compilation and setup
Once the prerequisites are setup, full compilation can be achieved using cmake. 
Starting from the root directory, run 
```
mkdir build
cmake ../
make -j4 install
cd ../
```
Finally, you need to make sure that the root directory is added to your python path
```
export PYTHONPATH=~/PATH/TO/QUANTEL:$PYTHONPATH
```
