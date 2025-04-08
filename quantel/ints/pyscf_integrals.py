#!/usr/bin/python

from quantel.utils.linalg import orthogonalisation_matrix
import pyscf
import numpy as np

class PySCFMolecule(pyscf.gto.Mole):
    """Wrapper class to call molecule functions from PySCF"""
    def __init__(self,_atom,_basis,_unit):
        """Initialise the PySCF molecule
                _atom  : str
                    The path to the atom file
                _basis : str
                    The basis set
                _unit  : str
                    The unit of the atom file (Angstrom or Bohr)
        """
        # Get spin from 2nd line of atom file
        with open(_atom) as f:
            f.readline()
            tmp = f.readline().split()
            _charge = int(tmp[0])
            _spin = int(tmp[1])-1
        # Initialise underlying PySCF molecule
        super().__init__(atom=_atom,basis=_basis,unit=_unit,spin=_spin,charge=_charge)
        self.atom = _atom
        self.unit = _unit
        self.build()

    def nalfa(self):
        """Return the number of alpha electrons"""
        return self.nelec[0]
    
    def nbeta(self):
        """Return the number of beta electrons"""
        return self.nelec[1]
    
    def natom(self):
        """Return the number of atoms"""
        return len(self.atom)
        
    def multiplicity(self):
        """Return the multiplicity of the molecule"""
        return self.spin + 1
    
    def print(self):
        """Print the molecular coordinates in the same way as quantel.LibintInterface"""
        for ID, coords in self._atom:
            print("  {:>4s}: {: 10.6f}  {: 10.6f}  {: 10.6f}".format(ID,coords[0],coords[1],coords[2]))


class PySCFIntegrals:
    """Wrapper class to call integral functions from PySCF"""
    #"MGGA_C_TPSS"
    def __init__(self,mol,xc=None,kscale=1.0):
        """ Initialise the PySCF interface from PySCF molecule
                mol : PySCFMolecule
                    The PySCF molecule object
                xc  : str
                    The exchange-correlation functional
        """
        self.mol = mol
        self.kscale = kscale
        
        # Initialise overlap matrix and orthogonalisation matrix
        self.S = self.mol.intor("int1e_ovlp")
        self.X = orthogonalisation_matrix(self.S)
        # Initialise one-electron integrals
        self.oei = self.mol.intor("int1e_kin") + self.mol.intor("int1e_nuc")
        self.xc = xc
        if(self.xc is not None):
            # Initialise the grid for numerical integration
            self.grid = pyscf.dft.gen_grid.Grids(self.mol)
            # Initialise numerical integration
            self.ni = pyscf.dft.numint.NumInt()
        
    def molecule(self):
        """Return the molecule object"""
        return self.mol

    def nbsf(self):
        """Return the number of basis functions"""
        return self.mol.nao
    
    def nmo(self):
        """Return the number of molecular orbitals"""
        return self.mol.nao
    
    def scalar_potential(self):
        """Return the nuclear repulsion energy"""
        return self.mol.energy_nuc()
    
    def overlap_matrix(self):
        """Return the overlap matrix"""
        return self.S
    
    def oei_matrix(self,spin=None):
        """Return the one-electron integrals"""
        return self.oei
    
    def orthogonalization_matrix(self):
        """Return the orthogonalisation matrix"""
        return self.X
    
    def build_fock(self,dm):
        """ Build the Fock matrix
            Args:
                dm : ndarray
                    The density matrix
            Returns:
                ndarray : The Fock matrix (h + 2 * J - K)
        """
        vJ, vK = pyscf.scf.hf.get_jk(self.mol, dm)
        return self.oei + 2 * vJ - vK
    
    def build_multiple_JK(self,vdJ,vdK,nj,nk):
        """ Build Coulomb and Exchange matrices for multiple sets of densities
            Args:
                vdJ : ndarray
                    The Coulomb matrices
                vdK : ndarray
                    The Exchange matrices
                nj  : int
                    The number of Coulomb densities
                nk  : int
                    The number of exchange densities
            Returns:
                ndarray : The Coulomb matrix
                ndarray : The Exchange matrix
        """
        vJ, vK = pyscf.scf.hf.get_jk(self.mol, (vdJ,vdK), hermi=0)
        return vJ[0], vK[1]
    
    def build_vxc(self,dma,dmb):
        """ Build the exchange-correlation potential
            Args:
                dma : ndarray
                    The alpha density matrix
                dmb : ndarray
                    The beta density matrix
            Returns:
                float : The exchange-correlation energy
                ndarray : The alpha exchange-correlation potential
                ndarray : The beta exchange-correlation potential   
        """
        n, exc, vxc = self.ni.nr_uks(self.mol, self.grid, self.xc, (dma,dmb))
        return exc, vxc[0], vxc[1]
    
    def oei_ao_to_mo(self, C1, C2, spin=None):
        """ Transform the one-electron integrals from AO to MO basis
            Args:
                C1 : ndarray
                    The alpha MO coefficients
                C2 : ndarray
                    The beta MO coefficients
            Returns:
                ndarray : The transformed one-electron integrals
        """
        return np.linalg.multi_dot([C1.T, self.oei, C1])

    def tei_array(self):
        """ Return an array containing the AO eri integrals"""
        n = self.nbsf()
        return np.reshape(self.mol.intor("int2e", aosym="s1"),(n,n,n,n))
    
    def tei_ao_to_mo(self, C1, C2, C3, C4, alpha1, alpha2):
        """ Transform the two-electron integrals from AO to MO basis. Order is <12|34> (physicists)
            Args:
                C1 : ndarray
                    The alpha MO coefficients
                C2 : ndarray
                    The beta MO coefficients
                C3 : ndarray
                    The alpha MO coefficients
                C4 : ndarray
                    The beta MO coefficients
                alpha1 : bool
                    Spin of the first electron
                alpha2 : bool
                    Spin of the second-electron
            Returns:
                ndarray : The transformed two-electron integrals
        """
        self.eri = self.mol.intor("int2e", aosym="s1")
        # NOTE, need to convert from physicists to chemists notation <12|34> = (13|24)
        mo_eri = pyscf.ao2mo.incore.general(self.eri, (C1,C3,C2,C4), compact=False)
        mo_eri = mo_eri.transpose(0,2,1,3)
        # Return antisymmetrised two-electron integrals as appropriate
        if(alpha1 == alpha2):
            return mo_eri - mo_eri.transpose(0,1,3,2)
        else:
            return mo_eri
