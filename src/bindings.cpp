#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <libint2/initialize.h>

#include "libint_interface.h"
#include "molecule.h"    
#include "determinant.h"
#include "mo_integrals.h"
#include "excitation.h"
#include "ci_space.h"
#include "ci_space2.h"
#include "four_array.h"
#include "two_array.h"

namespace py = pybind11;
using namespace pybind11::literals;

py::array_t<double,py::array::c_style> vec_to_np_array(size_t d1, double *data)
{
     py::array_t<double,py::array::c_style> array(d1,data);
     return array.reshape({d1});
}

py::array_t<double,py::array::c_style> vec_to_np_array(size_t d1, size_t d2, double *data)
{
     py::array_t<double,py::array::c_style> array(d1*d2,data);
     return array.reshape({d1,d2});
}

py::array_t<double,py::array::c_style> vec_to_np_array(size_t d1, size_t d2, size_t d3, double *data)
{
     py::array_t<double,py::array::c_style> array(d1*d2*d3,data);
     return array.reshape({d1,d2,d3});
}

py::array_t<double,py::array::c_style> vec_to_np_array(size_t d1, size_t d2, size_t d3, size_t d4, double *data)
{
     py::array_t<double,py::array::c_style> array(d1*d2*d3*d4, data);
     return array.reshape({d1,d2,d3,d4});
}

bool libint_initialize() 
{
     static bool initialized = false;
     if(initialized) {
          return true;
     }
     // Initialize libint2
     libint2::initialize();
     libint2::set_solid_harmonics_ordering(libint2::SHGShellOrdering_Gaussian);

     initialized = true;
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
          .def(py::init<std::string>())
          .def(py::init<std::vector<std::tuple<int,double,double,double>>,std::string>())
          .def(py::init<std::vector<std::tuple<std::string,double,double,double>>,std::string>())
          .def(py::init<std::string,std::string>())
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

     py::class_<Determinant>(m, "Determinant")
          .def(py::init<>(), "Default constructor")
          .def(py::init<std::vector<uint8_t>, std::vector<uint8_t> >(), "Constructor with occupation vectors")
          .def(py::init<std::string> (), "Constructor from determinant string")
          .def("apply_excitation", py::overload_cast<Eph &, bool>(&Determinant::apply_excitation), "Apply single excitation operator")
          .def("apply_excitation", py::overload_cast<Epphh &, bool, bool>(&Determinant::apply_excitation), "Apply double excitation operator")
          .def("__lt__", &Determinant::operator<, "Comparison operator");

     py::class_<Eph>(m, "Eph").def(py::init<size_t,size_t>(), "Constructor with indices");
     py::class_<Epphh>(m, "Epphh").def(py::init<size_t,size_t,size_t,size_t>(), "Constructor with indices");

     py::class_<CIspace>(m, "CIspace")
          .def(py::init<MOintegrals &,size_t,size_t,size_t>(), "Constructor with number of electrons and orbitals")
          .def("initialize", [](CIspace &self, std::string citype, std::vector<std::string> detlist)
               {
                    self.initialize(citype, detlist);
               },py::arg("citype"), py::arg("detlist") = std::vector<std::string>(),
               "Initialize the CI space")
          .def("print", &CIspace::print, "Print the CI space")
          .def("print_vector", &CIspace::print_vector, "Print a CI vector")
          .def("ndet", &CIspace::ndet, "Get the number of determinants")
          .def("nalfa", &CIspace::nalfa, "Get the number of high-spin electrons")
          .def("nbeta", &CIspace::nbeta, "Get the number of low-spin electrons")
          .def("nmo", &CIspace::nmo, "Get the number of molecular orbitals")
          .def("get_det_index", &CIspace::get_det_index, "Get the index of a determinant")
          .def("get_det_list", &CIspace::get_det_list, "Get the list of determinants")
          .def("H_on_vec", [](CIspace &ci, py::array_t<double> &V)
               {
                    size_t ndet = ci.ndet();
                    auto Vbuf = V.request();
                    std::vector<double> v_V((double *) Vbuf.ptr, (double *) Vbuf.ptr + Vbuf.size);
                    std::vector<double> sigma(ndet, 0.0);
                    ci.H_on_vec(v_V, sigma);
                    return vec_to_np_array(ndet, sigma.data());
               },
               "Compute the one-electron part of the sigma vector")
          .def("build_Hmat", [](CIspace &ci) 
               {
                    size_t ndet = ci.ndet();
                    std::vector<double> Hmat(ndet*ndet,0.0);
                    ci.build_Hmat(Hmat);
                    return vec_to_np_array(ndet,ndet,Hmat.data());
               },
               "Build the Hamiltonian matrix")
          .def("build_Hd", [](CIspace &ci) 
               {
                    size_t ndet = ci.ndet();
                    std::vector<double> Hdiag(ndet,0.0);
                    ci.build_Hd(Hdiag);
                    return vec_to_np_array(ndet,Hdiag.data());
               },
               "Build the Hamiltonian matrix")
          .def("trdm1", [](
               CIspace &ci, py::array_t<double> &bra, py::array_t<double> &ket, 
               bool alpha)
               {
                    size_t nmo = ci.nmo();
                    auto bra_buf = bra.request();
                    auto ket_buf = ket.request();
                    std::vector<double> v_bra(
                         (double *) bra_buf.ptr, (double *) bra_buf.ptr + bra_buf.size);
                    std::vector<double> v_ket(
                         (double *) ket_buf.ptr, (double *) ket_buf.ptr + ket_buf.size);
                    std::vector<double> rdm1(nmo*nmo,0.0);
                    ci.build_rdm1(v_bra,v_ket,rdm1,alpha);
                    return vec_to_np_array(nmo,nmo,rdm1.data());
               },
               "Build the one-particle transition reduced density matrix")
          .def("trdm2", [](
               CIspace &ci, py::array_t<double> &bra, py::array_t<double> &ket, 
               bool alpha1, bool alpha2)
               {
                    size_t nmo = ci.nmo();
                    auto bra_buf = bra.request();
                    auto ket_buf = ket.request();
                    std::vector<double> v_bra(
                         (double *) bra_buf.ptr, (double *) bra_buf.ptr + bra_buf.size);
                    std::vector<double> v_ket(
                         (double *) ket_buf.ptr, (double *) ket_buf.ptr + ket_buf.size);
                    std::vector<double> rdm2(nmo*nmo*nmo*nmo,0.0);
                    ci.build_rdm2(v_bra,v_ket,rdm2,alpha1,alpha2);
                    return vec_to_np_array(nmo,nmo,nmo,nmo,rdm2.data());
               },
               "Build the two-particle transition reduced density matrix")
          .def("rdm1", [](
               CIspace &ci, py::array_t<double> &ket, bool alpha)
               {
                    size_t nmo = ci.nmo();
                    auto ket_buf = ket.request();
                    std::vector<double> v_ket(
                         (double *) ket_buf.ptr, (double *) ket_buf.ptr + ket_buf.size);
                    std::vector<double> rdm1(nmo*nmo,0.0);
                    ci.build_rdm1(v_ket,v_ket,rdm1,alpha);
                    return vec_to_np_array(nmo,nmo,rdm1.data());
               },
               "Build the one-particle reduced density matrix")
          .def("rdm2", [](
               CIspace &ci, py::array_t<double> &ket, 
               bool alpha1, bool alpha2)
               {
                    size_t nmo = ci.nmo();
                    auto ket_buf = ket.request();
                    std::vector<double> v_ket(
                         (double *) ket_buf.ptr, (double *) ket_buf.ptr + ket_buf.size);
                    std::vector<double> rdm2(nmo*nmo*nmo*nmo,0.0);
                    ci.build_rdm2(v_ket,v_ket,rdm2,alpha1,alpha2);
                    return vec_to_np_array(nmo,nmo,nmo,nmo,rdm2.data());
               },
               "Build the two-particle reduced density matrix");      

     py::class_<CIspace2>(m, "CIspace2")
          .def(py::init<MOintegrals &,size_t,size_t,size_t>(), "Constructor with number of electrons and orbitals")
          .def("initialize", [](CIspace2 &self, std::string citype, std::vector<std::string> detlist)
               {
                    self.initialize(citype, detlist);
               },py::arg("citype"), py::arg("detlist") = std::vector<std::string>(),
               "Initialize the CI space")
          .def("print", &CIspace2::print, "Print the CI space")
          .def("print_vector", &CIspace2::print_vector, "Print a CI vector")
          .def("ndet", &CIspace2::ndet, "Get the number of determinants")
          .def("nalfa", &CIspace2::nalfa, "Get the number of high-spin electrons")
          .def("nbeta", &CIspace2::nbeta, "Get the number of low-spin electrons")
          .def("nmo", &CIspace2::nmo, "Get the number of molecular orbitals")
          .def("get_det_index", &CIspace2::get_det_index, "Get the index of a determinant")
          .def("get_det_list", &CIspace2::get_det_list, "Get the list of determinants")
          .def("H_on_vec", [](CIspace2 &ci, py::array_t<double> &V)
               {
                    size_t ndet = ci.ndet();
                    auto Vbuf = V.request();
                    std::vector<double> v_V((double *) Vbuf.ptr, (double *) Vbuf.ptr + Vbuf.size);
                    std::vector<double> sigma(ndet, 0.0);
                    ci.H_on_vec(v_V, sigma);
                    return vec_to_np_array(ndet, sigma.data());
               },
               "Compute the one-electron part of the sigma vector")
          .def("build_Hmat", [](CIspace2 &ci) 
               {
                    size_t ndet = ci.ndet();
                    std::vector<double> Hmat(ndet*ndet,0.0);
                    ci.build_Hmat(Hmat);
                    return vec_to_np_array(ndet,ndet,Hmat.data());
               },
               "Build the Hamiltonian matrix")
          .def("build_Hd", [](CIspace2 &ci) 
               {
                    size_t ndet = ci.ndet();
                    std::vector<double> Hdiag(ndet,0.0);
                    ci.build_Hd(Hdiag);
                    return vec_to_np_array(ndet,Hdiag.data());
               },
               "Build the Hamiltonian matrix")
          //.def("trdm1", [](
          //     CIspace2 &ci, py::array_t<double> &bra, py::array_t<double> &ket, 
          //     bool alpha)
          //     {
          //          size_t nmo = ci.nmo();
          //          auto bra_buf = bra.request();
          //          auto ket_buf = ket.request();
          //          std::vector<double> v_bra(
          //               (double *) bra_buf.ptr, (double *) bra_buf.ptr + bra_buf.size);
          //          std::vector<double> v_ket(
          //               (double *) ket_buf.ptr, (double *) ket_buf.ptr + ket_buf.size);
          //          std::vector<double> rdm1(nmo*nmo,0.0);
          //          ci.build_rdm1(v_bra,v_ket,rdm1,alpha);
          //          return vec_to_np_array(nmo,nmo,rdm1.data());
          //     },
          //     "Build the one-particle transition reduced density matrix")
          /*.def("trdm2", [](
               CIspace2 &ci, py::array_t<double> &bra, py::array_t<double> &ket, 
               bool alpha1, bool alpha2)
               {
                    size_t nmo = ci.nmo();
                    auto bra_buf = bra.request();
                    auto ket_buf = ket.request();
                    std::vector<double> v_bra(
                         (double *) bra_buf.ptr, (double *) bra_buf.ptr + bra_buf.size);
                    std::vector<double> v_ket(
                         (double *) ket_buf.ptr, (double *) ket_buf.ptr + ket_buf.size);
                    std::vector<double> rdm2(nmo*nmo*nmo*nmo,0.0);
                    ci.build_rdm2(v_bra,v_ket,rdm2,alpha1,alpha2);
                    return vec_to_np_array(nmo,nmo,nmo,nmo,rdm2.data());
               },
               "Build the two-particle transition reduced density matrix")
          .def("rdm1", [](
               CIspace2 &ci, py::array_t<double> &ket, bool alpha)
               {
                    size_t nmo = ci.nmo();
                    auto ket_buf = ket.request();
                    std::vector<double> v_ket(
                         (double *) ket_buf.ptr, (double *) ket_buf.ptr + ket_buf.size);
                    std::vector<double> rdm1(nmo*nmo,0.0);
                    ci.build_rdm1(v_ket,v_ket,rdm1,alpha);
                    return vec_to_np_array(nmo,nmo,rdm1.data());
               },
               "Build the one-particle reduced density matrix")
          .def("rdm2", [](
               CIspace2 &ci, py::array_t<double> &ket, 
               bool alpha1, bool alpha2)
               {
                    size_t nmo = ci.nmo();
                    auto ket_buf = ket.request();
                    std::vector<double> v_ket(
                         (double *) ket_buf.ptr, (double *) ket_buf.ptr + ket_buf.size);
                    std::vector<double> rdm2(nmo*nmo*nmo*nmo,0.0);
                    ci.build_rdm2(v_ket,v_ket,rdm2,alpha1,alpha2);
                    return vec_to_np_array(nmo,nmo,nmo,nmo,rdm2.data());
               },
               "Build the two-particle reduced density matrix")
          */
               ;      

     py::class_<MOintegrals>(m, "MOintegrals")
          .def(py::init<double,TwoArray &,FourArray &,size_t,double>(),
               py::arg("scalar_potential"),py::arg("oei"), py::arg("tei"), py::arg("nmo"), 
               py::arg("tolerance")=1e-14,
               "Initialise MO integrals from TwoArray/FourArray objects")
          .def(py::init([](double scalar_pot,
                           py::array_t<double, py::array::c_style> oei_arr,
                           py::array_t<double, py::array::c_style> tei_arr,
                           size_t nmo,
                           double tolerance = 1e-14)
               {
                    // oei: 2D array
                    auto oei_buf = oei_arr.request();
                    if (oei_buf.ndim != 2) throw std::runtime_error("oei must be a 2D numpy array");
                    std::vector<double> v_oei((double*)oei_buf.ptr, (double*)oei_buf.ptr + oei_buf.size);
                    TwoArray h1e(v_oei,oei_buf.shape[0],oei_buf.shape[1]);

                    // tei: 4D array
                    auto tei_buf = tei_arr.request();
                    if (tei_buf.ndim != 4) throw std::runtime_error("tei must be a 4D numpy array");
                    std::vector<double> v_tei((double*)tei_buf.ptr, (double*)tei_buf.ptr + tei_buf.size);
                    FourArray eri(v_tei,tei_buf.shape[0],tei_buf.shape[1],tei_buf.shape[2],tei_buf.shape[3]);

                    return new MOintegrals(scalar_pot,h1e,eri,nmo,tolerance);
               }),
               py::arg("scalar_potential"), py::arg("oei"), py::arg("tei"), py::arg("nmo"),
               py::arg("tolerance")=1e-14,
               "Initialise MO integrals from numpy arrays (oei: 2D, tei: 4D)")
          .def("scalar_potential", &MOintegrals::scalar_potential, "Get the value of the scalar potential")
          .def("oei", &MOintegrals::oei, py::arg("p"), py::arg("q"),
               "Get an element of the one-electron Hamiltonian matrix")
          .def("oei_eff", &MOintegrals::oei_eff, py::arg("p"), py::arg("q"),
               "Get an element of the effective 1e Hamiltonian matrix h_pq = hpq - 0.5 * sum_r <pr|rq>")
          .def("tei", &MOintegrals::tei, py::arg("p"), py::arg("q"), py::arg("r"), py::arg("s"),
               "Get an element of the two-electron integrals <pq||rs>")
          .def("oei_matrix", &MOintegrals::oei_matrix, "Get a pointer to the one-electron Hamiltonian matrix")
          .def("tei_array", &MOintegrals::tei_array, "Get a pointer to the two-electron integral array")
          .def("nmo", &MOintegrals::nmo, "Get the number of molecular orbitals");

//     py::class_<MOintegrals>(m, "MOintegrals")
//          .def(py::init<LibintInterface &>(), "Initialise MO integrals from LibintInterface object")
//          .def("update_orbitals",[](MOintegrals &m_ints, py::array_t<double>  C, size_t ninactive, size_t nactive) 
//               {
//                    auto C_buf = C.request();
///                    std::vector<double> v_C((double *) C_buf.ptr, (double *) C_buf.ptr + C_buf.size);
//                    m_ints.update_orbitals(v_C,ninactive,nactive);
//               },
//               "Compute MO integrals from MO coefficients")
//          .def("scalar_potential", &MOintegrals::scalar_potential, "Get the value of the scalar potential")
//          .def("oei_matrix", [](MOintegrals &mo_ints, bool alpha) 
//               { 
//                    size_t nact = mo_ints.nact();
//                    return vec_to_np_array(nact,nact,mo_ints.oei_matrix(alpha)); 
//               }, 
//               "Return one-electron Hamiltonian matrix in MO basis")
//          .def("tei_array", [](MOintegrals &mo_ints, bool alpha1, bool alpha2) 
//               { 
//                    size_t nact = mo_ints.nact();
//                    return vec_to_np_array(nact,nact,nact,nact,mo_ints.tei_array(alpha1, alpha2)); 
//               },
//               "Return two-electron integral array")
//          .def("nbsf", &MOintegrals::nbsf, "Get the number of basis functions")
//          .def("nmo", &MOintegrals::nmo, "Get the number of molecular orbitals")
//          .def("nact", &MOintegrals::nact, "Get the number of active orbitals")
//          .def("ncore", &MOintegrals::ncore, "Get the number of inactive orbitals")
//          ;
          
     py::class_<FourArray>(m, "FourArray")
          .def(py::init([](py::array_t<double, py::array::c_style> data) {
               // Get dimensions
               auto buf = data.request();
               if (buf.ndim != 4)
                   throw std::runtime_error("Input array must have 4 dimensions");
               size_t dim1 = buf.shape[0];
               size_t dim2 = buf.shape[1];
               size_t dim3 = buf.shape[2];
               size_t dim4 = buf.shape[3];
               std::vector<double> v_data((double*)buf.ptr, (double*)buf.ptr + buf.size);
               return new FourArray(v_data, dim1, dim2, dim3, dim4);
            }), py::arg("data"), "Constructor for FourArray accepting a numpy array as data")
          .def("dim", &FourArray::dim, "Get dimensions of the FourArray")
          .def("__call__", [](FourArray &self, size_t p, size_t q, size_t r, size_t s) -> double {
                  return self(p, q, r, s);
            }, py::arg("p"), py::arg("q"), py::arg("r"), py::arg("s"), "Access element via obj(p,q,r,s)")
            .def("__getitem__", [](FourArray &self, py::tuple idx) -> double {
                  if (idx.size() != 4)
                         throw py::index_error("FourIndexArray indices must be a 4-tuple");
                  size_t p = idx[0].cast<size_t>();
                  size_t q = idx[1].cast<size_t>();
                  size_t r = idx[2].cast<size_t>();
                  size_t s = idx[3].cast<size_t>();
                  return self(p, q, r, s);
            }, "Access element via obj[p, q, r, s]")
          .def("array", [](FourArray &self) {
               size_t d1, d2, d3, d4;
               std::tie(d1, d2, d3, d4) = self.dim();
               return vec_to_np_array(d1, d2, d3, d4, &self(0,0,0,0));
               },
               "Return the underlying data as a numpy array")
          ;

     py::class_<TwoArray>(m, "TwoArray")
          .def(py::init([](py::array_t<double, py::array::c_style> data) 
               {
                    // Get dimensions
                    auto buf = data.request();
                    if (buf.ndim != 2)
                        throw std::runtime_error("Input array must have 2 dimensions");
                    size_t dim1 = buf.shape[0];
                    size_t dim2 = buf.shape[1];
                    std::vector<double> v_data((double*)buf.ptr, (double*)buf.ptr + buf.size);
                    return new TwoArray(v_data, dim1, dim2);
               }), py::arg("data"), "Constructor for TwoArray accepting a numpy array as data")
          .def("dim", &TwoArray::dim, "Get dimensions of the TwoArray")
          .def("__call__", [](TwoArray &self, size_t p, size_t q) -> double 
               {
                    return self(p, q);
               }, py::arg("p"), py::arg("q"), "Access element via obj(p,q)")
          .def("__getitem__", [](TwoArray &self, py::tuple idx) -> double 
               {
                  if (idx.size() != 2)
                         throw py::index_error("TwoArray indices must be a 2-tuple");
                  size_t p = idx[0].cast<size_t>();
                  size_t q = idx[1].cast<size_t>();
                  return self(p, q);
               }, "Access element via obj[p, q]")
          .def("array", [](TwoArray &self) {
               size_t d1, d2;
               std::tie(d1, d2) = self.dim();
               return vec_to_np_array(d1, d2, &self(0,0)); 
               },
               "Return the underlying data as a numpy array")
          ;

     py::class_<LibintInterface>(m, "LibintInterface")
          .def(py::init<const std::string, Molecule &>())
          .def("initalize", &LibintInterface::initialize, "Initialize matrix elements")
          .def("nbsf", &LibintInterface::nbsf, "Get number of basis functions")
          .def("nmo", &LibintInterface::nmo, "Get number of molecular orbitals")
          .def("scalar_potential", &LibintInterface::scalar_potential, "Get value of the scalar potential")
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
               std::vector<double> v_jk(v_dens.size(),0.0);
               ints.build_JK(v_dens, v_jk);
               return vec_to_np_array(nbsf,nbsf,v_jk.data()); 
               },
               "Build JK matrix from density matrix")
          .def("build_multiple_JK", [](LibintInterface &ints, 
                    py::array_t<double> &vDJ, py::array_t<double> &vDK, size_t nj, size_t nk) {
               size_t nbsf = ints.nbsf();
               auto vDJ_buf = vDJ.request();
               auto vDK_buf = vDK.request();
               std::vector<double> v_vDJ((double *) vDJ_buf.ptr, (double *) vDJ_buf.ptr + vDJ_buf.size);
               std::vector<double> v_vDK((double *) vDK_buf.ptr, (double *) vDK_buf.ptr + vDK_buf.size);
               std::vector<double> v_J(nbsf*nbsf*nj,0.0);
               std::vector<double> v_K(nbsf*nbsf*nk,0.0);
               ints.build_multiple_JK(v_vDJ,v_vDK,v_J,v_K,nj,nk);
               return std::make_tuple(vec_to_np_array(nj,nbsf,nbsf,v_J.data()), vec_to_np_array(nk,nbsf,nbsf,v_K.data()));
               },
               "Build J and K matrices from a list of density matrices"
          )
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
               "Return one-electron Hamiltonian matrix")
          .def("dipole_matrix", [](LibintInterface &ints) { 
               return vec_to_np_array(4, ints.nbsf(), ints.nbsf(), ints.dipole_integrals()); 
               },
               "Return the dipole matrix integrals")
          .def("mo_integrals", [](LibintInterface &ints, 
               py::array_t<double> &C, size_t ncore, size_t nactive) 
               {
                    size_t nbsf = ints.nbsf();
                    auto C_buf = C.request();
                    std::vector<double> v_C((double *) C_buf.ptr, (double *) C_buf.ptr + C_buf.size);
                    return ints.mo_integrals(v_C, ncore, nactive);
               },
               "Compute MO integrals from MO coefficients"
          )
          .def("tei_ao_to_mo", [](LibintInterface &ints, 
               py::array_t<double> &C1, py::array_t<double> &C2, 
               py::array_t<double> &C3, py::array_t<double> &C4, 
               bool alpha1, bool alpha2) 
               {
                    size_t nbsf = ints.nbsf();
                    // Get the buffer for the numpy arrays
                    auto C1_buf = C1.request();
                    auto C2_buf = C2.request();
                    auto C3_buf = C3.request();
                    auto C4_buf = C4.request();
                    // Get the dimensions of the transformation matrices
                    size_t d1 = C1_buf.shape[1];
                    size_t d2 = C2_buf.shape[1];
                    size_t d3 = C3_buf.shape[1];
                    size_t d4 = C4_buf.shape[1];
                    // Get the data from the numpy arrays
                    std::vector<double> v_C1((double *) C1_buf.ptr, (double *) C1_buf.ptr + C1_buf.size);
                    std::vector<double> v_C2((double *) C2_buf.ptr, (double *) C2_buf.ptr + C2_buf.size);
                    std::vector<double> v_C3((double *) C3_buf.ptr, (double *) C3_buf.ptr + C3_buf.size);
                    std::vector<double> v_C4((double *) C4_buf.ptr, (double *) C4_buf.ptr + C4_buf.size);
                    // Allocate memory for the MO integrals
                    std::vector<double> v_eri(d1*d2*d3*d4, 0.0);
                    // Perform the transformation
                    ints.tei_ao_to_mo(v_C1,v_C2,v_C3,v_C4,v_eri,alpha1,alpha2);
                    // Return the MO integrals as a numpy array
                    return vec_to_np_array(d1,d2,d3,d4,v_eri.data()); 
               },
               "Perform AO to MO transformation"
          )
          .def("oei_ao_to_mo", [](LibintInterface &ints, 
               py::array_t<double> &C1, py::array_t<double> &C2, bool alpha) 
               {
                    size_t nbsf = ints.nbsf();
                    // Get the buffer for the numpy arrays
                    auto C1_buf = C1.request();
                    auto C2_buf = C2.request();
                    // Get the dimensions of the transformation matrices
                    size_t d1 = C1_buf.shape[1];
                    size_t d2 = C2_buf.shape[1];
                    // Get the data from the numpy arrays
                    std::vector<double> v_C1((double *) C1_buf.ptr, (double *) C1_buf.ptr + C1_buf.size);
                    std::vector<double> v_C2((double *) C2_buf.ptr, (double *) C2_buf.ptr + C2_buf.size);
                    // Allocate memory for the MO integrals
                    std::vector<double> v_oei(d1*d2, 0.0);
                    // Perform the transformation
                    ints.oei_ao_to_mo(v_C1,v_C2,v_oei,alpha);
                    // Return the MO integrals as a numpy array
                    return vec_to_np_array(d1,d2,v_oei.data()); 
               },
               "Perform AO to MO transformation for one-electron integrals"
          )
          .def("tei_array", [](LibintInterface &ints) { 
               size_t nbsf = ints.nbsf();
               return vec_to_np_array(nbsf, nbsf, nbsf, nbsf, ints.tei_array());
               },
               "Return two-electron integral (pq|rs) array");


     m.def("det_str", &det_str,"Print the determinant");

}
