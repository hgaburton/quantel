#!/usr/bin/python

from quantel.utils.linalg import orthogonalisation_matrix
import pyscf
import numpy as np
import scipy.special
from pyscf import fci

class PySCFMolecule(pyscf.gto.Mole):
    """Wrapper class to call molecule functions from PySCF"""
    def __init__(self,_atom,_basis,_unit,charge=0,spin=0):
        """Initialise the PySCF molecule
                _atom  : str
                    The path to the atom file
                _basis : str
                    The basis set
                _unit  : str
                    The unit of the atom file (Angstrom or Bohr)
                _charge : int
                    Charge of molecule (default = 0)
                _spin  : int
                    Spin ms value (defualt = 0)
        """
        # Get spin from 2nd line of atom file
        # Initialise underlying PySCF molecule
        super().__init__(atom=_atom,basis=_basis,unit=_unit,spin=spin,charge=charge)
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
            # Get range-separation and hybrid coefficients
            self.omega, self.alpha, self.hybrid_K = self.ni.rsh_and_hybrid_coeff(self.xc)
        else:
            self.grid = None
            self.ni = None
            self.omega = 0.0
            self.alpha = 0.0
            self.hybrid_K = 1.0
        
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
    
    def dipole_matrix(self,origin=None):
        """Return the dipole matrix"""
        if origin is None:
            origin = np.zeros(3)
        else:
            origin = np.asarray(origin, dtype=np.float64)

        # Get nuclear dipole    
        charges = self.mol.atom_charges()
        coords  = self.mol.atom_coords()
        nucl_dip = np.einsum('i,ix->x',charges,coords - origin[None,:])

        # Get AO dipole matrices
        ao_dip = self.mol.intor_symmetric('int1e_r', comp=3)
        return nucl_dip, ao_dip 

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
    
    def build_multiple_JK(self,vdJ,vdK):
        """ Build Coulomb and Exchange matrices for multiple sets of densities
            Args:
                vdJ : ndarray
                    The Coulomb matrices
                vdK : ndarray
                    The Exchange matrices
            Returns:
                ndarray : The Coulomb matrix
                ndarray : The Exchange matrix
        """
        if(self.xc is None):
            vJ, vK = pyscf.scf.hf.get_jk(self.mol, (vdJ,vdK), hermi=1)
            return vJ[0], vK[1], vK[1]
        
        if(not self.ni.libxc.is_hybrid_xc(self.xc)):
            vJ, vK = pyscf.scf.hf.get_jk(self.mol, (vdJ,vdK), hermi=1)
            vKfunc = np.zeros_like(vK)
        else:

            vJ, vK = pyscf.scf.hf.get_jk(self.mol, (vdJ,vdK), hermi=1)
            if(self.omega == 0):
                vKfunc = vK * self.hybrid_K
            elif self.alpha == 0: # LR=0, only SR exchange
                _, vKfunc = pyscf.scf.hf.get_jk(self.mol, (vdJ,vdK), hermi=1, omega=-self.omega, with_j=False)
                vKfunc *= self.hybrid_K
            elif self.hybrid_K == 0: # SR=0, only LR exchange
                _, vKfunc = pyscf.scf.hf.get_jk(self.mol, (vdJ,vdK), hermi=1, omega=self.omega, with_j=False)
                vKfunc *= (self.alpha)
            else: # SR and LR different ratios
                _, vKlr = pyscf.scf.hf.get_jk(self.mol, (vdJ,vdK), hermi=1, omega=self.omega, with_j=False)
                vKfunc = vKfunc * self.hybrid_K + (self.alpha - self.hybrid_K) * vKlr
        return vJ[0], vK[1], vKfunc[1]
    
    def build_JK(self,dm):
        """ Build the Coulomb and Exchange matrices
            Args:
                dm : ndarray
                    The density matrix
            Returns:
                ndarray : The Coulomb matrix
                ndarray : The Exchange matrix
        """
#        if not self.ni.libxc.is_hybrid_xc(self.xc):
        vJ, vK = self.build_multiple_JK(dm,dm)
        return 2 * vJ - vK
    
    def build_vxc(self,dms):
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
        _, exc, vxc = self.ni.nr_uks(self.mol, self.grid, self.xc, dms)
        return exc, vxc
    
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

    def tei_array(self,spin1=None,spin2=None):
        """ Return an array containing the AO eri integrals"""
        n = self.nbsf()
        return np.reshape(self.mol.intor("int2e", aosym="s1"),(n,n,n,n))#.transpose(0,2,1,3)
    
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
        
    def cache_xc_kernel(self, mo_coeff, mo_occ, spin):
        """ Cache the exchange-correlation kernel for given densities
            Args:
                dma : ndarray
                    The alpha density matrix
                dmb : ndarray
                    The beta density matrix
        """
        return self.ni.cache_xc_kernel(self.mol, self.grid, self.xc, mo_coeff, mo_occ, spin)
    
    def uks_fxc(self, dm1, rho0, vxc, fxc):
        """ Compute the fxc kernel for first-order density dm1 with precomputed rho0, vxc, fxc.
            Args:
                dm1 : 2 x ndarray
                    The first-order density matrix [dm1a,dm1b]
                rho0 : ndarray
                    The zeroth-order density on the grid (used to define the kernel)
                vxc : ndarray
                    The exchange-correlation potential on the grid
                fxc : ndarray
                    The exchange-correlation kernel on the grid
            Returns:
                ndarray : The fxc contribution to the Fock matrix as [alpha,beta]
        """
        return self.ni.nr_uks_fxc(self.mol,self.grid,self.xc,None,dm1,0,hermi=0,rho0=rho0,vxc=vxc,fxc=fxc)

class PySCF_MO_Integrals:
    """Wrapper class to call integral functions from PySCF"""
    def __init__(self, ints):
        """ Initialise the PySCF interface from PySCF molecule
                ints : PySCFIntegrals
                    The PySCF integrals object
        """
        # Save the integral object
        self.ints = ints
        self.Vnuc = self.ints.scalar_potential()
        # Initialise number of active orbitals to 0
        self.m_nact = 0

    def nbsf(self):
        """Return the number of basis functions"""
        return self.ints.nbsf()

    def nmo(self):
        """Return the number of molecular orbitals"""
        return self.ints.nmo()
    
    def nact(self):
        """Return the number of active orbitals"""
        return self.m_nact
    
    def scalar_potential(self):
        """Return the nuclear repulsion energy"""
        return self.m_V
    
    def oei_matrix(self, alfa1):
        """ Return the one-electron integrals in MO basis
            Args:
                alfa1 : int
                    The spin of the electron
            Returns:
                ndarray : The one-electron integrals in MO basis
        """
        return self.oei_a if alfa1 else self.oei_b
    
    def tei_array(self, alfa1, alfa2):
        """ Return the two-electron integrals in MO basis
            Args:
                alfa1 : int
                    The spin of the first electron
                alfa2 : int
                    The spin of the second electron
            Returns:
                ndarray : The two-electron integrals in MO basis
        """
        if(alfa1 and alfa2):
            return self.tei_aa
        elif(alfa1 and not alfa2):
            return self.tei_ab
        elif((not alfa1) and (not alfa2)):
            return self.tei_bb
        else:
            raise ValueError("Invalid spin combination")
        
    def compute_core_potential(self):
        """ Compute the core potential
            Returns:
                ndarray : The core potential
        """
        if(self.m_ncore > 0):
            # Get core coefficients
            self.m_Ccore = self.m_C[:,:self.m_ncore]
            # Compute core density
            self.m_Pcore = self.m_Ccore @ self.m_Ccore.T
            # Compute inactive JK matrix (2J-K) in AO basis
            JK = self.ints.build_JK(self.m_Pcore)

            # Compute scalar core energy 
            Hao = self.ints.oei_matrix()
            self.Vc = 2 * np.einsum('pq,pq', Hao + 0.5 * JK, self.m_Pcore)

            # Compute core potential in MO basis
            self.Vc_oei = self.m_Cact.T @ (JK @ self.m_Cact)
        else:
            self.Vc = 0.0
            self.Vc_oei = np.zeros((self.m_nact,self.m_nact))

    def compute_scalar_potential(self):
        """ Compute the scalar potential """
        self.m_V = self.ints.scalar_potential() + self.Vc

    def compute_oei(self, alpha):
        """ Compute the one-electron integrals
            Args:
                alpha : bool
                    The spin of the electron
        """
        hao = self.ints.oei_matrix(alpha)
        hmo = np.linalg.multi_dot([self.m_Cact.T, hao, self.m_Cact])
        if(alpha): self.oei_a = hmo + self.Vc_oei
        else: self.oei_b = hmo + self.Vc_oei

    def compute_tei(self, alpha1, alpha2):
        """ Compute the two-electron integrals
            Args:
                alpha1 : bool
                    The spin of the first electron
                alpha2 : bool
                    The spin of the second electron
        """
        # Get the active MO coefficients
        C = self.m_Cact
        if(alpha1 and alpha2):
            self.tei_aa = self.ints.tei_ao_to_mo(C,C,C,C,alpha1,alpha2)
        elif(alpha1 and not alpha2):
            self.tei_ab = self.ints.tei_ao_to_mo(C,C,C,C,alpha1,alpha2)
        elif((not alpha1) and (not alpha2)):
            self.tei_bb = self.ints.tei_ao_to_mo(C,C,C,C,alpha1,alpha2)

    def update_orbitals(self, C, ncore, nactive):
        """ Update the active orbitals and integrals
            Args:
                C : ndarray
                    The MO coefficients
                ncore : int
                    The number of core orbitals
                nactive : int
                    The number of active orbitals
        """
        self.m_ncore = ncore
        self.m_nact  = nactive
        self.m_C = C.copy()
        self.m_Cact = C[:,ncore:ncore+nactive]

        self.compute_core_potential()
        self.compute_scalar_potential()
        self.compute_oei(True)
        self.compute_oei(False)
        self.compute_tei(True,True)
        self.compute_tei(True,False)
        self.compute_tei(False,True)

class PySCF_CIspace:
    """Class to compute the CI space for a given molecule"""
    def __init__(self, mo_ints, nmo, nalfa, nbeta):
        """
        Initialise the CI space
            Args:
                mo_ints : PySCF_MO_Integrals
                    The PySCF integrals object
                nmo     : int
                    The number of molecular orbitals
                nalfa   : int
                    The number of alpha electrons
                nbeta   : int
                    The number of beta electrons
        """
        # Save the integrals object
        self.m_ints = mo_ints
        self.m_nmo = nmo
        self.m_nalfa = nalfa
        self.m_nbeta = nbeta
        self.nelec = (nalfa, nbeta)
        # Initialise FCI solver
        self.fcisolver = pyscf.fci.direct_spin1


    def ndeta(self):
        return scipy.special.comb(self.m_nmo,self.m_nalfa).astype(int)
    def ndetb(self):
        return scipy.special.comb(self.m_nmo,self.m_nbeta).astype(int)
    def ndet(self):
        return self.ndeta() * self.ndetb()
    def nalfa(self):
        return self.m_nalfa
    def nbeta(self):
        return self.m_nbeta
    
    def build_Hmat(self):
        """ Build the CI Hamiltonian matrix
            Returns:
                ndarray : The CI Hamiltonian matrix
        """
        h1 = self.m_ints.oei_matrix(True)
        h2 = self.m_ints.tei_array(True,False).transpose(0,2,1,3)
        return self.fcisolver.pspace(h1,h2,self.m_nmo,self.nelec)[1] + np.eye(self.ndet()) * self.m_ints.scalar_potential()
    
    def rdm1(self, civec, spin):
        """ Compute the one-particle density matrix
            Args:
                civec : ndarray
                    The CI vector
            Returns:
                ndarray : The one-particle density matrix
        """
        rdm1a, rdm1b = self.fcisolver.make_rdm1s(civec,self.m_nmo,self.nelec)
        if(spin): return rdm1a
        else: return rdm1b
            
    def rdm2(self, civec, spin1, spin2):
        """ Compute the two-particle density matrix
            Args:
                civec : ndarray
                    The CI vector
            Returns:
                ndarray : The two-particle density matrix
        """
        dm2aa, dm2ab, dm2bb = self.fcisolver.make_rdm12s(civec,self.m_nmo,self.nelec)[1]
        if(spin1 and spin2): return dm2aa.transpose(0,2,1,3)
        if(spin1 and not spin2): return dm2ab.transpose(0,2,1,3)
        if(not spin1 and not spin2): return dm2bb.transpose(0,2,1,3)

    def trdm1(self, cibra, ciket, spin):
        """ Compute the transition one-particle density matrix
        """
        trdm1a, trdm1b = self.fcisolver.trans_rdm1s(cibra,ciket,self.m_nmo,self.nelec)
        if(spin): return trdm1a
        else: return trdm1b

    def trdm2(self, cibra, ciket,  spin1, spin2):
        """ Compute the transition one-particle density matrix
        """
        tdm2aa, tdm2ab, tdm2ba, tdm2bb = self.fcisolver.trans_rdm12s(cibra,ciket,self.m_nmo,self.nelec)[1]
        if(spin1 and spin2): return tdm2aa.transpose(0,2,1,3)
        if(spin1 and not spin2): return tdm2ab.transpose(0,2,1,3)
        if(not spin1 and not spin2): return tdm2bb.transpose(0,2,1,3)
