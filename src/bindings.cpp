#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <libint2/initialize.h>

#include "libint_interface.h"
#include "molecule.h"

namespace py = pybind11;
using namespace pybind11::literals;

py::array_t<double,py::array::c_style> vec_to_np_array(size_t m, size_t n, double *data)
{
     py::array_t<double,py::array::c_style> array(m*n,data);
     return array.reshape({m,n});
}

bool libint_initialize() 
{
     static bool initialized = false;
     if(initialized) {
          return true;
     }
     // Initialize libint2
     libint2::initialize();

     initialized = true;
     std::cout << "Libint2 initialized successfully" << std::endl;
     return true;
}

void libint_finalize() 
{
     libint2::finalize();
}

PYBIND11_MODULE(_quantel, m) {
     m.def("libint_initialize", &libint_initialize, "Called upon to initalise module");
     m.def("libint_finalize", &libint_finalize, "Called upon to finalise on module exit");

     py::class_<Molecule>(m, "Molecule")
          .def(py::init<>())
          .def(py::init<std::vector<std::tuple<int,double,double,double> > >())
          .def(py::init<std::vector<std::tuple<std::string,double,double,double> > >())
          .def("add_atom", py::overload_cast<int,double,double,double>(&Molecule::add_atom), 
               "Add an atom using integer nuclear charge")
          .def("add_atom", py::overload_cast<std::string,double,double,double>(&Molecule::add_atom), 
               "Add an atom using element string")
          .def("set_charge", &Molecule::set_charge, "Set molecular charge")
          .def("set_spin_multiplicity", py::overload_cast<size_t>(&Molecule::set_spin_multiplicity), 
               "Set molecular spin multiplicity")
          .def("set_spin_multiplicity", py::overload_cast<>(&Molecule::set_spin_multiplicity), 
               "Set default molecular spin multiplicity")
          .def("print", &Molecule::print, "Print the molecular structure")
          .def("natom", &Molecule::natom, "Get the number of atoms")
          .def("nelec", &Molecule::nelec, "Get the number of electrons")
          .def("nalfa", &Molecule::nalfa, "Get the number of high-spin electrons")
          .def("nbeta", &Molecule::nbeta, "Get the number of low-spin electrons")
          .def("charge", &Molecule::charge, "Get the total charge")
          .def("mult", &Molecule::mult, "Get the spin multiplicity");
          

     py::class_<LibintInterface>(m, "LibintInterface")
          .def(py::init<const std::string, Molecule &>())
          .def("initalize", &LibintInterface::initialize, "Initialize matrix elements")
          .def("nbsf", &LibintInterface::nbsf, "Get number of basis functions")
          .def("nmo", &LibintInterface::nmo, "Get number of molecular orbitals")
          .def("scalar_potential", &LibintInterface::scalar_potential, "Get value of the scalar potential")
          .def("overlap", &LibintInterface::overlap, "Get element of overlap matrix")
          .def("oei", &LibintInterface::oei, "Get element of one-electron Hamiltonian matrix")
          .def("tei", &LibintInterface::tei, "Get element of two-electron integral array")
          .def("molecule", &LibintInterface::molecule, "Get molecule object")
          .def("build_fock", [](LibintInterface &ints, py::array_t<double> &dens) {
               size_t nbsf = ints.nbsf();
               auto dens_buf = dens.request();
               std::vector<double> v_dens((double *) dens_buf.ptr, (double *) dens_buf.ptr + dens_buf.size);
               std::vector<double> v_fock(v_dens.size(), 0.0);
               ints.build_fock(v_dens, v_fock);
               return vec_to_np_array(nbsf,nbsf,v_fock.data()); 
               },
               "Build Fock matrix from density matrix")
          .def("build_JK", [](LibintInterface &ints, py::array_t<double> &dens) {
               size_t nbsf = ints.nbsf();
               auto dens_buf = dens.request();
               std::vector<double> v_dens((double *) dens_buf.ptr, (double *) dens_buf.ptr + dens_buf.size);
               std::vector<double> v_jk(v_dens.size(), 0.0);
               ints.build_fock(v_dens, v_jk);
               return vec_to_np_array(nbsf,nbsf,v_jk.data()); 
               },
               "Build JK matrix from density matrix")
          .def("overlap_matrix", [](LibintInterface &ints) { 
               return vec_to_np_array(ints.nbsf(), ints.nbsf(), ints.overlap_matrix()); 
               },
               "Return the overlap matrix")
          .def("orthogonalization_matrix", [](LibintInterface &ints) { 
               return vec_to_np_array(ints.nbsf(), ints.nbsf(), ints.orthogonalization_matrix()); 
               },
               "Return the orthogonalization matrix")
          .def("oei_matrix", [](LibintInterface &ints, bool alpha) { 
               return vec_to_np_array(ints.nbsf(), ints.nbsf(), ints.oei_matrix(alpha)); 
               }, 
               "Return one-electron Hamiltonian matrix");
}
