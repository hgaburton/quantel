
#include <fmt/core.h>
#include "molecule.h"
#include "periodic_table.h"

void Molecule::add_atom(int nuc_charge, double x, double y, double z)
{
    // Create the atom
    libint2::Atom atom;
    atom.atomic_number = nuc_charge;
    atom.x = x; atom.y = y; atom.z = z;
    // Add to the atom vector
    atoms.push_back(atom);
    m_natom++;
}

void Molecule::add_atom(std::string element, double x, double y, double z)
{ 
    // Transform element label to uppercase
    std::transform(element.begin(), element.end(), element.begin(),
                   [](unsigned char c){ return std::toupper(c); });
    // Add atom using atomic charge from peridoic_table dict
    add_atom(periodic_table.at(element), x, y, z);
}

void Molecule::print()
{
    for(size_t iatm=0; iatm<m_natom; iatm++)
    {
        double x = atoms[iatm].x;
        double y = atoms[iatm].y;
        double z = atoms[iatm].z;
        int nuc_charge = atoms[iatm].atomic_number;
        std::string str = fmt::format("{:>4s}: {: 10.6f}  {: 10.6f}  {: 10.6f}\n", 
                                      element_labels.at(nuc_charge), x, y, z);
        fmt::print(str);
    }
}