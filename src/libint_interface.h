#ifndef LIBINT_INTERFACE_H
#define LIBINT_INTERFACE_H

#include <libint2.hpp>
#include "integral_interface.h"
#include "molecule.h"

// Inherit from IntegralInterface
class LibintInterface : public IntegralInterface {
    /// \brief LibintInterface class
    /// \details This class provides an interface to the Libint2 library for computing molecular integrals
 
private:
    const libint2::BasisSet m_basis; //!< The Libint2 basis set
    const Molecule &m_mol;

public:

    /** \brief Destructor for the interface
     **/
    virtual ~LibintInterface() { }

    /** \brief Constructor for the interface
     **/
    LibintInterface(const std::string basis_str, Molecule &mol) :
        m_mol(mol), m_basis(basis_str, mol.atoms)
    { 
        init(m_basis.nbf(),m_mol.nalfa(),m_mol.nbeta()), 
        initialize();
    }

    /// Return pointer to m_mol
    const Molecule &molecule() { return m_mol; }

    /// Initialise all relevant variables
    void initialize();

    /// Plot orbitals
    void molden_orbs(std::vector<double> &C, 
                     std::vector<double> &occ, 
                     std::vector<double> &evals);

private:
    /// Compute the nuclear repulsion energy
    void compute_nuclear_potential();
    /// Compute overlap integrals
    void compute_overlap();
    /// Compute the one-electron Hamiltonian integrals
    void compute_one_electron_matrix();
    /// Compute the two-electron integrals
    void compute_two_electron_integrals();
    /// Compute the dipole integrals
    void compute_dipole_integrals();
};

#endif // LIBINT_INTERFACE_H
