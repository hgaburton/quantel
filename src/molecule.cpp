#include <fmt/core.h>
#include "molecule.h"
#include "periodic_table.h"

const double ANG_TO_BOHR = 1.8897261254578281;

Molecule::Molecule(std::vector<std::tuple<int,double,double,double> > _atoms, std::string units) :
    m_units(units)
{
    for(size_t iatm=0; iatm < _atoms.size(); iatm++)
    {
        int nuc_charge = std::get<0>(_atoms[iatm]);
        double x = std::get<1>(_atoms[iatm]);
        double y = std::get<2>(_atoms[iatm]);
        double z = std::get<3>(_atoms[iatm]);
        add_atom(nuc_charge, x, y, z);
    }

    // Update the spin multiplicity
    set_spin_multiplicity();
}

// Initialise molecule from xyz file with element labels
Molecule::Molecule(std::string fname, std::string units) :
    m_units(units)
{
    // Open the file
    std::ifstream file(fname);
    if(!file.is_open())
    {
        throw std::runtime_error("Could not open file");
    }

    // Read the number of atoms
    size_t natom, charge, mult;
    file >> natom;
    // Read the charge and multiplicity
    file >> charge >> mult;

    // Read the atoms
    for(size_t iatm=0; iatm<natom; iatm++)
    {
        std::string element;
        double x, y, z;
        file >> element >> x >> y >> z;
        add_atom(element, x, y, z);
    }

    // Close the file
    file.close();

    // Update the charge and spin multiplicity
    set_spin_multiplicity(mult);
    set_charge(charge);
}

Molecule::Molecule(std::vector<std::tuple<std::string,double,double,double> > _atoms, std::string units) :
    m_units(units)
{
    for(size_t iatm=0; iatm < _atoms.size(); iatm++)
    {
        std::string element = std::get<0>(_atoms[iatm]);
        double x = std::get<1>(_atoms[iatm]);
        double y = std::get<2>(_atoms[iatm]);
        double z = std::get<3>(_atoms[iatm]);
        add_atom(element, x, y, z);
    }

    // Update the spin multiplicity
    set_spin_multiplicity();
}

void Molecule::add_atom(int nuc_charge, double x, double y, double z)
{
    // Create the atom
    libint2::Atom atom;
    atom.atomic_number = nuc_charge;
    if(m_units == "bohr")
    {
        atom.x = x;
        atom.y = y;
        atom.z = z;
    }
    else if(m_units == "angstrom")
    {
        atom.x = x * ANG_TO_BOHR;
        atom.y = y * ANG_TO_BOHR;
        atom.z = z * ANG_TO_BOHR;
    }
    else
    {
        throw std::runtime_error("Invalid units. Must be 'bohr' or 'angstrom'");
    }

    // Add to the atom vector
    atoms.push_back(atom);

    // Increment the atom and electron count
    m_natom++;
    m_nelec += nuc_charge;
}

void Molecule::add_atom(std::string element, double x, double y, double z)
{ 
    // Transform element label to uppercase
    std::transform(element.begin(), element.end(), element.begin(),
                   [](unsigned char c){ return std::toupper(c); });
    // Add atom using atomic charge from peridoic_table dict
    add_atom(periodic_table.at(element), x, y, z);
}

void Molecule::set_charge(int charge)
{
    // Set the charge
    m_charge = charge;
    m_nelec  = m_nelec - m_charge;

    // Update the spin multiplicity
    set_spin_multiplicity();
}

void Molecule::set_spin_multiplicity()
{
    // Default multiplicity, assume maximal pairing
    set_spin_multiplicity(m_nelec%2+1);   
}

void Molecule::set_spin_multiplicity(size_t mult)
{
    // Check if the multiplicity is valid
    if((m_nelec + mult - 1)%2 != 0)
    {   
        std::cout << "N_electron: " << m_nelec << " Multiplicity: " << mult << std::endl;
        throw std::runtime_error("Invalid spin multiplicity");
    }

    // Set the spin multiplicity
    m_mult = mult;
    m_nalfa = (m_nelec + m_mult - 1) / 2;
    m_nbeta = m_nelec - m_nalfa;
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