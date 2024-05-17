#ifndef MOLECULE_H
#define MOLECULE_H

#include <libint2.hpp>
#include <vector>


class Molecule {
protected:
    size_t m_natom; //!< Number of atoms
    
    int m_charge = 0; //!< Total charge
    size_t m_mult; //!< Spin multiplicity

    size_t m_nelec = 0; //!< Number of electrons
    size_t m_nalfa = 0; //!< Number of alpha electrons
    size_t m_nbeta = 0; //!< Number of beta electrons

public:
    std::vector<libint2::Atom> atoms; //!< Vector of atomic positions

    /** \brief Destructor **/
    virtual ~Molecule() { }

    /** \brief Default constructor **/
    Molecule() { m_natom = 0; }

    /// \brief Create molecule from list of atom tuples
    /// \param atoms List of tuples containing atomic number and coordinates
    Molecule(std::vector<std::tuple<int,double,double,double> > _atoms);

    /// \brief Create molecule from list of atom tuples
    /// \param atoms List of tuples containing element symbol and coordinates
    Molecule(std::vector<std::tuple<std::string,double,double,double> > _atoms);

    /// \brief Add an atom to the molecule using nuclear charge
    /// \param nuc_charge Nuclear atomic charge
    /// \param x x-coordinate
    /// \param y y-coordinate
    /// \param z z-coordinate
    virtual void add_atom(int nuc_charge, double x, double y, double z);

    /// \brief Add an atom to the molecule using element symbol
    /// \param element String containing element name
    /// \param x x-coordinate
    /// \param y y-coordinate
    /// \param z z-coordinate
    virtual void add_atom(std::string element, double x, double y, double z);

    /// \brief Set molecular spin multiplicity
    /// \param mult Spin multiplicity
    virtual void set_spin_multiplicity(size_t mult);

    /// \brief Set default molecular spin multiplicity
    virtual void set_spin_multiplicity();

    /// \brief Set molecular charge
    /// \param charge Total charge of the molecule
    virtual void set_charge(int charge);

    /// \brief Print the molecular structure
    virtual void print();

    /// \brief Get the number of atoms
    /// \return Number of atoms
    size_t natom() const { return m_natom; }

    /// \brief Get the total charge
    /// \return Total charge
    size_t charge() const { return m_charge; }

    /// \brief Get the spin multiplicity
    /// \return Spin multiplicity
    size_t mult() const { return m_mult; }

    /// \brief Get the number of electrons
    /// \return Number of electrons
    size_t nelec() const { return m_nelec; }

    /// \brief Get the number of alpha electrons
    /// \return Number of alpha electrons
    size_t nalfa() const { return m_nalfa; }

    /// \brief Get the number of beta electrons
    /// \return Number of beta electrons
    size_t nbeta() const { return m_nbeta; }
};

#endif