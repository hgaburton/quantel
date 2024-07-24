#!/bin/bash

mkdir -p external
cd external

echo "Collecting fmt library..."
git clone https://github.com/fmtlib/fmt.git 1> /dev/null

echo "Collecting libint2 source..."
wget https://github.com/evaleev/libint/releases/download/v2.9.0/libint-2.9.0.tgz 1> /dev/null
tar -xvf libint-2.9.0.tgz 1> /dev/null
mv libint-2.9.0 libint2 1> /dev/null

echo "Collecting armadillo source..."
wget https://sourceforge.net/projects/arma/files/armadillo-12.8.3.tar.xz 1> /dev/null
tar -xvf armadillo-12.8.3.tar.xz 1> /dev/null
mv armadillo-12.8.3 armadillo 1> /dev/null

cd ../

echo "Setup build directory and compile..."
rm -r build &> /dev/null
mkdir -p build
cd build
cmake ../
make -j4 install
cd ../
