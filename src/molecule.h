#ifndef MOLECULE_H
#define MOLECULE_H

#include <libint2.hpp>
#include <vector>


class Molecule {
protected:
    int m_natom; //!< Number of atoms

public:
    std::vector<libint2::Atom> atoms; //!< Vector of atomic positions

    /** \brief Destructor **/
    virtual ~Molecule() { }

    /** \brief Default constructor **/
    Molecule() : 
        m_natom(0)
    { }

    /** \brief Add an atom to the molecule using nuclear charge
        \param nuc_charge Nuclear atomic charge
        \param x x-coordinate
        \param y y-coordinate
        \param z z-coordinate
     **/
    virtual void add_atom(int nuc_charge, double x, double y, double z);

    /** \brief Add an atom to the molecule using element symbol
        \param element String containing element name
        \param x x-coordinate
        \param y y-coordinate
        \param z z-coordinate
     **/
    virtual void add_atom(std::string element, double x, double y, double z);

    /** \brief Print the molecular structure **/
    virtual void print();
};

#endif