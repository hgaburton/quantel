#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include "test.h"


namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(quantel, m) {
    m.def("add", &add, "Function that adds two numbers");
}