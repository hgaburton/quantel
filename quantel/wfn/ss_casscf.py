#!/usr/bin/python3
# Author: Antoine Marie, Hugh G. A. Burton

import numpy as np
import scipy.special
import h5py # type: ignore
import quantel
from functools import reduce
from quantel.utils.linalg import orthogonalise, matrix_print
from quantel.gnme.cas_noci import cas_coupling
from .wavefunction import Wavefunction
from quantel.ints.pyscf_integrals import PySCF_MO_Integrals, PySCF_CIspace


class SS_CASSCF(Wavefunction):
    """
        State-specific CASSCF object

        Inherits from the Wavefunction abstract base class with the pure virtual 
        properties:
           - energy
           - gradient
           - hessian
           - take_step
           - save_last_step
           - restore_step
    """
    def __init__(self, integrals, active_space, ncore=None, verbose=0):
        """ Initialise the SS-CASSCF object
                integrals: quantel integral interface
                active_space: tuple of active space definition (nmo, (nalfa, nbeta))
                ncore: number of core electrons
                verbose: verbosity level
        """
        # Initialise integrals object
        self.integrals  = integrals
        self.nalfa      = integrals.molecule().nalfa()
        self.nbeta      = integrals.molecule().nbeta()
        # Get number of basis functions and linearly independent orbitals
        self.nbsf       = integrals.nbsf()
        self.nmo        = integrals.nmo()
        # Get active space definition
        self.cas_nmo    = active_space[0]
        if(type(active_space[1]) is int):
            self.cas_nbeta = active_space[1]//2
            self.cas_nalfa = active_space[1] - self.cas_nbeta
        else:
            self.cas_nalfa  = active_space[1][0]
            self.cas_nbeta  = active_space[1][1]

        # Get number of core electrons
        if ncore is None:
            ne = integrals.molecule().nalfa() + integrals.molecule().nbeta()
            self.ncore = ne - self.cas_nalfa - self.cas_nbeta
            if(self.ncore % 2 != 0):
                raise ValueError("Number of core electrons must be even")
            if(self.ncore < 0):
                raise ValueError("Number of core electrons must be positive")
            self.ncore = self.ncore // 2
        else:
            self.ncore = ncore
        # Number of "occupied" orbitals
        self.nocc       = self.ncore + self.cas_nmo
        self.sanity_check()

        # Initialise mo integrals and CI space
        if(type(integrals) is quantel.ints.pyscf_integrals.PySCFIntegrals):
            self.mo_ints    = PySCF_MO_Integrals(integrals)
            self.cispace    = PySCF_CIspace(self.mo_ints, self.cas_nmo, self.cas_nalfa, self.cas_nbeta)
        else:
            self.mo_ints    = quantel.MOintegrals(integrals)
            self.cispace    = quantel.CIspace(self.mo_ints, self.cas_nmo, self.cas_nalfa, self.cas_nbeta)
            self.cispace.initialize('FCI')

        # Get number of determinants
        self.ndeta      = (scipy.special.comb(self.cas_nmo,self.cas_nalfa)).astype(int)
        self.ndetb      = (scipy.special.comb(self.cas_nmo,self.cas_nbeta)).astype(int)
        self.ndet       = (self.ndeta*self.ndetb).astype(int)

        # Save mapping indices for unique orbital rotations
        self.frozen     = None
        self.rot_idx    = self.uniq_var_indices(self.frozen)
        self.nrot       = np.sum(self.rot_idx)
 
    def sanity_check(self):
        '''Need to be run at the start of the kernel to verify that the number of 
           orbitals and electrons in the CAS are consistent with the system '''
        # Check number of active orbitals is positive
        if self.cas_nmo <= 0:
            raise ValueError("Number of active orbitals must be positive")
        # Check number of active electrons is positive
        if(self.cas_nalfa < 0):
            raise ValueError("Number of active alpha electrons must be >= 0")
        if(self.cas_nbeta < 0):
            raise ValueError("Number of active beta electrons must be >= 0")
        # Check number of active electrons doesn't exceed total number of electrons
        if(self.cas_nalfa > self.cas_nmo or self.cas_nbeta > self.cas_nmo):
            raise ValueError("Number of active electrons must be <= number of active orbitals")
        # Check number of active electrons doesn't exceed total number of electrons
        if(self.cas_nalfa > self.nalfa or self.cas_nbeta > self.nbeta):
            raise ValueError("Number of active electrons must be <= total number of electrons")
        # Check number of occupied orbitals doesn't exceed total number of orbitals
        if(self.nocc > self.nmo):
            raise ValueError("Number of inactive and active orbitals must be <= total number of orbitals")
       
    @property
    def dim(self):
        """Number of variables"""
        return self.nrot + self.ndet - 1

    @property
    def energy(self):
        """Energy of the current wavefunction"""
        E  = self.energy_core
        E += np.einsum('pq,pq', self.h1eff, self.dm1_cas, optimize="optimal")
        E += 0.5 * np.einsum('pqrs,pqrs', self.h2eff, self.dm2_cas, optimize="optimal")
        return E

    @property
    def sz(self):
        """<S_z> value of the current wave function"""
        return 0.5*(self.cas_nalfa - self.cas_nbeta)
    
    @property
    def s2(self):
        """ <S^2> value of the current wave function
            Uses the formula S^2 = S- S+ + Sz Sz + Sz, which corresponds to 
                <S^2> = <Sz> * (<Sz> + 1) + <Nb> - sum_pq G^{ab}_{pqqp} 
            where G^{ab}_{pqqp} is the alfa-beta component of the 2-RDM
        """
        civec  = self.mat_ci[:,0].copy()
        rdm2ab = self.cispace.rdm2(civec,True,False).transpose(0,2,1,3)
        return abs(self.sz * (self.sz + 1)  + self.cas_nbeta - np.einsum('pqqp',rdm2ab))

    @property
    def dipole(self):
        '''Compute the dipole vector'''
        ncore, nocc = self.ncore, self.ncore + self.cas_nmo
        # Get the dipole integrals
        nucl_dip, ao_dip = self.integrals.dipole_matrix()
        # Transform dipole matrices to MO basis
        dip_mo = np.einsum('xpq,pm,qn->xmn', ao_dip, self.mo_coeff, self.mo_coeff)
        # Nuclear and core contributions
        dip = nucl_dip + 2*np.einsum('ipp->i', dip_mo[:,:ncore,:ncore]) 
        # Active space contribution
        dip += np.einsum('ipq,pq->i', dip_mo[:,ncore:nocc,ncore:nocc], self.dm1_cas)
        return dip

    @property
    def quadrupole(self):
        '''Compute the quadrupole tensor'''
        raise NotImplementedError("Quadrupole computation not implemented")
        #ncore, nocc = self.ncore, self.ncore + self.ncas
        # Transform dipole matrices to MO basis
        #quad_mo = np.einsum('xypq,pm,qn->xymn', self.quad_mat, self.mo_coeff, self.mo_coeff)
        # Nuclear and core contributions
        #quad = self.quad_nuc.copy() 
        #quad += 2*np.einsum('xypp->xy', quad_mo[:,:,:ncore,:ncore]) 
        # Active space contribution
        #quad += np.einsum('xypq,pq->xy', quad_mo[:,:,ncore:nocc,ncore:nocc], self.dm1_cas)
        #return quad

    @property
    def gradient(self):
        """Compute the gradient of the energy with respect to the orbital and CI coefficients"""
        g_orb = self.get_orbital_gradient()
        g_ci  = self.get_ci_gradient()  
        return np.concatenate((g_orb, g_ci))

    @property
    def hessian(self):
        ''' This method concatenate the orb-orb, orb-CI and CI-CI part of the Hessian '''
        H_OrbOrb = (self.get_hessianOrbOrb()[:,:,self.rot_idx])[self.rot_idx,:]
        H_CICI   = self.get_hessianCICI()
        H_OrbCI  = self.get_hessianOrbCI()[self.rot_idx,:]
        return np.block([[H_OrbOrb, H_OrbCI],
                         [H_OrbCI.T, H_CICI]])

    
    def print(self,verbose=1):
        """ Print details about the state energy and orbital coefficients

            Inputs:
                verbose : level of verbosity
                          0 = No output
                          1 = Print energy components and spinao2mo
                          2 = Print energy components, spin, and exchange matrices
                          3 = Print energy components, spin, exchange matrices, and occupied orbital coefficients
                          4 = Print energy components, spin, exchange matrices, and all orbital coefficients
                          5 = Print energy components, spin, exchange matrices, generalised Fock matrix, and all orbital coefficients 
        """
        print("verbose = ", verbose)
        if(verbose > 0):
            print("\n ---------------------------------------------")
            print(f"         Total Energy = {self.energy:14.8f} Eh")
            print(" ---------------------------------------------")
            print(f"        <Sz> = {self.sz:5.2f}")
            print(f"        <S2> = {self.s2:5.2f}")
    
        if(verbose > 1):
            matrix_print(self.mo_coeff[:,:self.nocc], title="Occupied Orbital Coefficients")
        if(verbose > 2):
            matrix_print(self.mat_ci[:,[0]], title="CI Coefficients")
        if(verbose > 3):
            matrix_print(self.mo_coeff[:,self.nocc:], title="Virtual Orbital Coefficients", offset=self.nocc)
        if(verbose > 4):
            matrix_print(self.gen_fock[:self.nocc,:].T, title="Generalised Fock Matrix (MO basis)")
        print()

    def save_to_disk(self,tag):
        """Save a SS-CASSCF object to disk with prefix 'tag'"""
        # Canonicalise 
        #self.canonicalize()
        
        # Save hdf5 file with MO coefficients, orbital energies, energy, and spin
        with h5py.File(tag+".hdf5", "w") as F:
            F.create_dataset("mo_coeff", data=self.mo_coeff[:,:self.nocc])
            F.create_dataset("mat_ci", data=self.mat_ci[:,[0]])
            F.create_dataset("energy", data=self.energy)
            F.create_dataset("s2", data=self.s2)
        
        # Save numpy txt file with energy and Hessian indices
        hindices = self.get_hessian_index()
        with open(tag+".solution", "w") as F:
            F.write(f"{self.energy:18.12f} {hindices[0]:5d} {hindices[1]:5d} {self.s2:12.6f}\n")
        return

    def read_from_disk(self,tag):
        """Read a SS-CASSCF object from disk with prefix 'tag'"""
        # Read orbital and CI coefficients
        with h5py.File(tag+".hdf5", "r") as F:
            mo_read = F["mo_coeff"][:]
            ci_read = F["mat_ci"][:]
        # Check the input
        if mo_read.shape[0] != self.nbsf:
            raise ValueError("Inccorect number of AO basis functions in file")
        if ci_read.shape[0] != self.ndet:
            raise ValueError("Incorrect dimension of CI space in file")
        if mo_read.shape[1] < self.nocc:
            raise ValueError("Insufficient orbitals in file to represent occupied orbitals")
        if mo_read.shape[1] > self.nmo:
            raise ValueError("Too many orbitals in file")
        if ci_read.shape[1] > self.ndet:
            raise ValueError("Too many CI vectors in file")
        # Initialise the wave function
        self.initialise(mo_read, ci_read)
        return

    def copy(self, integrals=False):
        # Return a copy of the current object
        newcas = SS_CASSCF(self.integrals, [self.cas_nmo, [self.cas_nalfa, self.cas_nbeta]])
        newcas.initialise(self.mo_coeff, self.mat_ci, integrals=integrals)
        return newcas

    def overlap(self, them):
        """Compute the many-body overlap with another CAS waveunction (them)"""
        ovlp = self.integrals.overlap_matrix()
        return cas_coupling(self, them, ovlp)[0]

    def hamiltonian(self, them):
        """Compute the many-body Hamiltonian coupling with another CSF wavefunction (them)"""
        n2 = self.nbsf * self.nbsf 
        hcore = self.integrals.oei_matrix()
        eri   = self.integrals.tei_array().reshape(n2,n2)
        ovlp  = self.integrals.overlap_matrix()
        enuc  = self.integrals.scalar_potential()
        return cas_coupling(self, them, ovlp, hcore, eri, enuc)

    def tdm(self, them):
        """Compute the transition dipole moment with other CSF wave function (them)"""
        raise NotImplementedError("Transition dipole moment computation not implemented")



    def initialise(self, mo_guess, ci_guess, integrals=True):
        """ Initialise the SS-CASSCF object with a set of orbital and CI coefficients
                mo_guess:  initial guess for the orbital coefficients
                ci_guess:  initial guess for the CI coefficients
                integrals: flag to update integrals with the new coefficients
        """
        # Save orbital coefficients
        mo_guess = orthogonalise(mo_guess, self.integrals.overlap_matrix())
        self.mo_coeff = mo_guess
        self.nmo = self.mo_coeff.shape[1]
 
        # Save CI coefficients
        ci_guess = orthogonalise(ci_guess, np.identity(self.ndet)) 
        self.mat_ci = ci_guess

        # Initialise integrals
        if(integrals): self.update_integrals()

    def deallocate(self):
        """Deallocate member variables to save memory"""
        # Reduce the memory footprint for storing 
        self.ppoo   = None
        self.popo   = None
        self.h1e    = None
        self.h1eff  = None
        self.h2eff  = None
        self.F_core = None
        self.F_cas  = None
        self.ham    = None

    def guess_casci(self, n):
        """ Set initial guess to the CASCI state with index n"""
        self.mat_ci = np.linalg.eigh(self.ham)[1]
        self.mat_ci[:,[0,n]] = self.mat_ci[:,[n,0]]
        return 

    def update_integrals(self):
        """Update integrals with current set of orbital coefficients"""
        # Update integral interface
        self.mo_ints.update_orbitals(self.mo_coeff,self.ncore,self.cas_nmo)
        
        # Scalar energy (including core energy)
        self.energy_core = self.mo_ints.scalar_potential()
        # Effective one-electron Hamiltonian in CAS space
        self.h1eff = self.mo_ints.oei_matrix(True)
        # Two-electron ERI (transposed from phys -> chem notation)
        self.h2eff = self.mo_ints.tei_array(True,False).transpose(0,2,1,3)

        # Construct 1-electron integrals outside active space
        self.h1e = np.linalg.multi_dot([self.mo_coeff.T, self.integrals.oei_matrix(True), self.mo_coeff])
        # Construct required ERIs outside active space
        Cocc = self.mo_coeff[:,:self.nocc].copy()
        self.ppoo = self.integrals.tei_ao_to_mo(self.mo_coeff,Cocc,self.mo_coeff,Cocc,True,False).transpose(0,2,1,3)
        self.popo = self.integrals.tei_ao_to_mo(self.mo_coeff,self.mo_coeff,Cocc,Cocc,True,False).transpose(0,2,1,3)
        # Construct core potential outside active space
        dm_core = np.dot(self.mo_coeff[:,:self.ncore], self.mo_coeff[:,:self.ncore].T)
        v_jk = self.integrals.build_JK(dm_core)
        self.vhf_c = np.linalg.multi_dot([self.mo_coeff.T, v_jk, self.mo_coeff])

        # Reduced density matrices 
        self.dm1_cas, self.dm2_cas = self.get_casrdm_12()
        # Fock matrices
        self.get_fock_matrices()

        # Hamiltonian in active space
        self.ham = self.cispace.build_Hmat()
        # Sigma vector in active space
        civec    = self.mat_ci[:,0].copy()
        self.sigma = self.ham @ civec

        # Generalised Fock matrix
        self.gen_fock = self.get_gen_fock(self.dm1_cas, self.dm2_cas, False)
        return 

    def restore_last_step(self):
        """ Restore the coefficients to the last step"""
        self.mo_coeff = self.mo_coeff_save.copy()
        self.mat_ci   = self.mat_ci_save.copy()
        self.update_integrals()
        return

    def save_last_step(self):
        """ Save the current coefficients"""
        self.mo_coeff_save = self.mo_coeff.copy()
        self.mat_ci_save   = self.mat_ci.copy()
        return

    def take_step(self,step):
        """ Take a step in the combined orbital and CI space
            args:
                step: array of orbital and CI rotation angles
        """
        self.save_last_step()
        self.rotate_orb(step[:self.nrot])
        self.rotate_ci(step[self.nrot:])
        self.update_integrals()
        return

    def rotate_orb(self,step): 
        """ Rotate orbital coefficients by step using exponential transformation
            args:
                step: array of orbital rotation angles
        """
        orb_step = np.zeros((self.nmo,self.nmo))
        orb_step[self.rot_idx] = step
        self.mo_coeff = np.dot(self.mo_coeff, scipy.linalg.expm(orb_step - orb_step.T))
        return

    def rotate_ci(self,step): 
        """ Rotate CI coefficients by step using exponential transformation
            args:
                step: array of CI rotation angles
        """
        S       = np.zeros((self.ndet,self.ndet))
        S[1:,0] = step
        self.mat_ci = np.dot(self.mat_ci, scipy.linalg.expm(S - S.T))

    def transform_vector(self,vec,step,X):
        # No parallel transport as not yet implemented.
        return vec

    def get_casrdm_12(self):
        """ Compute the 1- and 2-electron reduced density matrices in the active space.
            returns:
                dm1_cas: 1-electron reduced density matrix
                dm2_cas: 2-electron reduced density matrix
        """
        # Get copy of CI vector
        civec = self.mat_ci[:,0].copy()
        # Make RDM1
        cas_rdm1a = self.cispace.rdm1(civec, True)
        cas_rdm1b = self.cispace.rdm1(civec, False)
        cas_rdm1 = cas_rdm1a + cas_rdm1b
        # Make RDM2 (need to convert phys -> chem notation)
        cas_rdm2aa = self.cispace.rdm2(civec, True, True).transpose(0,2,1,3)
        cas_rdm2bb = self.cispace.rdm2(civec, False, False).transpose(0,2,1,3)
        cas_rdm2ab = self.cispace.rdm2(civec, True, False).transpose(0,2,1,3)
        cas_rdm2 = cas_rdm2aa + cas_rdm2bb + cas_rdm2ab + cas_rdm2ab.transpose(2,3,0,1)
        return cas_rdm1, cas_rdm2
    
    def get_trdm12(self,ci1,ci2):
        """ Compute the transition density matrices between two CI vectors 
            args:
                ci1: CI vector for bra state
                ci2: CI vector for ket state
            returns:
                tdm1: 1-electron transition density matrix
                tdm2: 2-electron transition density matrix
        """
        # Transition 1RDM
        tdm1a = self.cispace.trdm1(ci1,ci2,True)
        tdm1b = self.cispace.trdm1(ci1,ci2,False)
        tdm1 = tdm1a + tdm1b
        # Transition 2RDM
        tdm2aa = self.cispace.trdm2(ci1,ci2,True,True).transpose(0,2,1,3)
        tdm2bb = self.cispace.trdm2(ci1,ci2,False,False).transpose(0,2,1,3)
        tdm2ab = self.cispace.trdm2(ci1,ci2,True,False).transpose(0,2,1,3)
        tdm2 = tdm2aa + tdm2bb + tdm2ab + tdm2ab.transpose(2,3,0,1)
        return tdm1, tdm2

    def get_spin_dm1(self):
        """ Compute the spin density in the AO basis
            returns:
                ao_spin_dens: spin density in the AO basis
        """
        # Get copy of CI vector
        civec = self.mat_ci[:,0].copy()
        # Make RDMs
        cas_rdm1a = self.cispace.rdm1(civec, True)
        cas_rdm1b = self.cispace.rdm1(civec, False)
        # Compute spin density
        spin_dens = cas_rdm1a - cas_rdm1b
        # Transform to AO basis
        mo_cas = self.mo_coeff[:,self.ncore:self.nocc]
        ao_spin_dens = np.linalg.multi_dot([mo_cas, spin_dens, mo_cas.T])
        return ao_spin_dens

    def get_fock_matrices(self):
        """ Compute the inactive and active Fock matrices"""
        # Core contribution is just effective 1-electron matrix
        self.F_core = self.h1e + self.vhf_c
          # Active space contribution
        self.F_cas = np.einsum('pq,mnpq->mn',self.dm1_cas,self.ppoo[:,:,self.ncore:,self.ncore:],optimize="optimal")
        self.F_cas -= 0.5 * np.einsum('pq,mqnp->mn',self.dm1_cas,self.popo[:,self.ncore:,:,self.ncore:],optimize="optimal")
        return

    def get_gen_fock(self,dm1_cas,dm2_cas,transition=False):
        """ Build generalised Fock matrix using 1- and 2-electron reduced density matrices
            args:
                dm1_cas: 1-electron reduced density matrix
                dm2_cas: 2-electron reduced density matrix
                transition: Flag to indicate generalised Fock matrix for transition density
        """
        ncore = self.ncore
        nocc  = self.nocc

        # Effective Coulomb and exchange operators for active space
        J_a = np.einsum('ij,pqij->pq', dm1_cas, self.ppoo[:,:,ncore:,ncore:],optimize="optimal")
        K_a = np.einsum('ij,piqj->pq', dm1_cas, self.popo[:,ncore:,:,ncore:],optimize="optimal")
        V_a = 2 * J_a - K_a

        # Universal contributions
        tdm1_cas = dm1_cas + dm1_cas.T
        F = np.zeros((self.nmo,self.nmo))
        if(not transition):
            F[:,:ncore] = 4.0 * self.F_core[:,:ncore] 
        F[:,ncore:nocc] += np.einsum('qx,yq->xy', self.F_core[ncore:nocc,:], tdm1_cas,optimize="optimal")
        F[:,:ncore] += V_a[:,:ncore] + (V_a[:ncore,:]).T

        # Effective interaction with orbitals outside active space
        tdm2_cas = dm2_cas + dm2_cas.transpose(1,0,2,3)
        ext_int  = np.einsum('bprs,pars->ba', self.popo[:,ncore:,ncore:nocc,ncore:], tdm2_cas,optimize="optimal")
        F[:ncore,ncore:nocc] += ext_int[:ncore,:]
        F[nocc:,ncore:nocc]  += ext_int[nocc:,:]
 
        return F

    def get_orbital_gradient(self):
        """ Build the orbital rotation part of the gradient"""
        #g_orb = self.get_gen_fock(self.dm1_cas, self.dm2_cas, False)
        return (self.gen_fock - self.gen_fock.T)[self.rot_idx]

    def get_ci_gradient(self):
        """ Build the CI component of the gradient"""
        if(self.ndet > 1):
            return 2.0 * np.dot(self.mat_ci[:,1:].T, self.sigma)
        else:
            return np.zeros((0))

    def get_hessianOrbCI(self):
        """ Build the orbital-CI part of the hessian"""
        H_OCI = np.zeros((self.nmo,self.nmo,self.ndet-1))
        ci1 = self.mat_ci[:,0].copy()
        for k in range(1,self.ndet):
            # Get transition density matrices
            ci2 = self.mat_ci[:,k].copy()
            dm1_cas, dm2_cas = self.get_trdm12(ci1,ci2)

            # Get transition generalised Fock matrix
            F = self.get_gen_fock(dm1_cas,dm2_cas,True)
            # Save component
            H_OCI[:,:,k-1] = 2 * (F - F.T)
        return H_OCI

    def get_hessianOrbOrb(self):
        """ Build the orbital-orbital part of the hessian"""
        norb  = self.nmo
        ncore = self.ncore
        nocc  = self.nocc
        ncas  = self.cas_nmo
        nvir  = norb - nocc

        Htmp = np.zeros((norb,norb,norb,norb))
        F_tot = self.F_core + self.F_cas

        # Temporary identity matrices 
        id_cor = np.identity(ncore)
        id_vir = np.identity(nvir)
        id_cas = np.identity(ncas)

        #virtual-core virtual-core H_{ai,bj}
        if ncore>0 and nvir>0:
            aibj = self.popo[nocc:,:ncore,nocc:,:ncore]
            abij = self.ppoo[nocc:,nocc:,:ncore,:ncore]

            Htmp[nocc:,:ncore,nocc:,:ncore] = ( 4 * (4 * aibj - abij.transpose((0,2,1,3)) - aibj.transpose((0,3,2,1)))  
                                              + 4 * np.einsum('ij,ab->aibj', id_cor, F_tot[nocc:,nocc:],optimize="optimal") 
                                              - 4 * np.einsum('ab,ij->aibj', id_vir, F_tot[:ncore,:ncore],optimize="optimal") )

        #virtual-core virtual-active H_{ai,bt}
        if ncore>0 and nvir>0:
            aibv = self.popo[nocc:,:ncore,nocc:,ncore:nocc]
            avbi = self.popo[nocc:,ncore:nocc,nocc:,:ncore]
            abvi = self.ppoo[nocc:,nocc:,ncore:nocc,:ncore]

            Htmp[nocc:,:ncore,nocc:,ncore:nocc] = ( 2 * np.einsum('tv,aibv->aibt', self.dm1_cas, 4 * aibv - avbi.transpose((0,3,2,1)) - abvi.transpose((0,3,1,2)),optimize="optimal") 
                                                  - 1 * np.einsum('ab,tvxy,vixy ->aibt', id_vir, self.dm2_cas, self.ppoo[ncore:nocc, :ncore, ncore:nocc, ncore:nocc],optimize="optimal") 
                                                  - 2 * np.einsum('ab,ti->aibt', id_vir, F_tot[ncore:nocc, :ncore],optimize="optimal") 
                                                  - 1 * np.einsum('ab,tv,vi->aibt', id_vir, self.dm1_cas, self.F_core[ncore:nocc, :ncore],optimize="optimal") )

        #virtual-active virtual-core H_{bt,ai}
        if ncore>0 and nvir>0:
             Htmp[nocc:, ncore:nocc, nocc:, :ncore] = np.einsum('aibt->btai', Htmp[nocc:, :ncore, nocc:, ncore:nocc],optimize="optimal")

        #virtual-core active-core H_{ai,tj}
        if ncore>0 and nvir>0:
            aivj = self.ppoo[nocc:,:ncore,ncore:nocc,:ncore]
            avji = self.ppoo[nocc:,ncore:nocc,:ncore,:ncore]
            ajvi = self.ppoo[nocc:,:ncore,ncore:nocc,:ncore]

            Htmp[nocc:,:ncore,ncore:nocc,:ncore] = ( 2 * np.einsum('tv,aivj->aitj', (2 * id_cas - self.dm1_cas), 4 * aivj - avji.transpose((0,3,1,2)) - ajvi.transpose((0,3,2,1)),optimize="optimal") 
                                                   - 1 * np.einsum('ji,tvxy,avxy -> aitj', id_cor, self.dm2_cas, self.ppoo[nocc:,ncore:nocc,ncore:nocc,ncore:nocc],optimize="optimal") 
                                                   + 4 * np.einsum('ij,at-> aitj', id_cor, F_tot[nocc:, ncore:nocc],optimize="optimal") 
                                                   - 1 * np.einsum('ij,tv,av-> aitj', id_cor, self.dm1_cas, self.F_core[nocc:, ncore:nocc],optimize="optimal"))

        #active-core virtual-core H_{tj,ai}
        if ncore>0 and nvir>0:
            Htmp[ncore:nocc, :ncore, nocc:, :ncore] = np.einsum('aitj->tjai',Htmp[nocc:,:ncore,ncore:nocc,:ncore],optimize="optimal")

        #virtual-active virtual-active H_{at,bu}
        if nvir>0:
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc]  = ( 2 * np.einsum('tuvx,abvx->atbu', self.dm2_cas, self.ppoo[nocc:,nocc:,ncore:nocc,ncore:nocc],optimize="optimal") 
                                                          + 2 * np.einsum('txvu,axbv->atbu', self.dm2_cas, self.popo[nocc:,ncore:nocc,nocc:,ncore:nocc],optimize="optimal") 
                                                          + 2 * np.einsum('txuv,axbv->atbu', self.dm2_cas, self.popo[nocc:,ncore:nocc,nocc:,ncore:nocc],optimize="optimal") )
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc] -= ( 1 * np.einsum('ab,tvxy,uvxy->atbu', id_vir, self.dm2_cas, self.ppoo[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc],optimize="optimal") 
                                                          + 1 * np.einsum('ab,tv,uv->atbu', id_vir, self.dm1_cas, self.F_core[ncore:nocc,ncore:nocc],optimize="optimal"))
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc] -= ( 1 * np.einsum('ab,uvxy,tvxy->atbu', id_vir, self.dm2_cas, self.ppoo[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc],optimize="optimal") 
                                                          + 1 * np.einsum('ab,uv,tv->atbu', id_vir, self.dm1_cas, self.F_core[ncore:nocc,ncore:nocc],optimize="optimal"))
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc] +=   2 * np.einsum('tu,ab->atbu', self.dm1_cas, self.F_core[nocc:, nocc:],optimize="optimal")

        #active-core virtual-active H_{ti,au}
        if ncore>0 and nvir>0:
            avti = self.ppoo[nocc:, ncore:nocc, ncore:nocc, :ncore]
            aitv = self.ppoo[nocc:, :ncore, ncore:nocc, ncore:nocc]

            Htmp[ncore:nocc,:ncore,nocc:,ncore:nocc]  = (- 2 * np.einsum('tuvx,aivx->tiau', self.dm2_cas, self.ppoo[nocc:,:ncore,ncore:nocc,ncore:nocc],optimize="optimal") 
                                                         - 2 * np.einsum('tvux,axvi->tiau', self.dm2_cas, self.ppoo[nocc:,ncore:nocc,ncore:nocc,:ncore],optimize="optimal") 
                                                         - 2 * np.einsum('tvxu,axvi->tiau', self.dm2_cas, self.ppoo[nocc:,ncore:nocc,ncore:nocc,:ncore],optimize="optimal") )
            Htmp[ncore:nocc,:ncore,nocc:,ncore:nocc] += ( 2 * np.einsum('uv,avti->tiau', self.dm1_cas, 4 * avti - aitv.transpose((0,3,2,1)) - avti.transpose((0,2,1,3)),optimize="optimal" ) 
                                                        - 2 * np.einsum('tu,ai->tiau', self.dm1_cas, self.F_core[nocc:,:ncore],optimize="optimal") 
                                                        + 2 * np.einsum('tu,ai->tiau', id_cas, F_tot[nocc:,:ncore],optimize="optimal"))

            #virtual-active active-core  H_{au,ti}
            Htmp[nocc:,ncore:nocc,ncore:nocc,:ncore]  = np.einsum('auti->tiau', Htmp[ncore:nocc,:ncore,nocc:,ncore:nocc],optimize="optimal")

        #active-core active-core H_{ti,uj}
        if ncore>0:
            viuj = self.ppoo[ncore:nocc,:ncore,ncore:nocc,:ncore]
            uvij = self.ppoo[ncore:nocc,ncore:nocc,:ncore,:ncore]

            Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore]  = 4 * np.einsum('tv,viuj->tiuj', id_cas - self.dm1_cas, 4 * viuj - viuj.transpose((2,1,0,3)) - uvij.transpose((1,2,0,3)),optimize="optimal" )
            Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore] += ( 2 * np.einsum('utvx,vxij->tiuj', self.dm2_cas, self.ppoo[ncore:nocc,ncore:nocc,:ncore,:ncore],optimize="optimal") 
                                                         + 2 * np.einsum('uxvt,vixj->tiuj', self.dm2_cas, self.ppoo[ncore:nocc,:ncore,ncore:nocc,:ncore],optimize="optimal") 
                                                         + 2  *np.einsum('uxtv,vixj->tiuj', self.dm2_cas, self.ppoo[ncore:nocc,:ncore,ncore:nocc,:ncore],optimize="optimal") )
            Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore] += ( 2 * np.einsum('tu,ij->tiuj', self.dm1_cas, self.F_core[:ncore, :ncore],optimize="optimal") 
                                                         - 2 * np.einsum('ij,tvxy,uvxy->tiuj', id_cor, self.dm2_cas, self.ppoo[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc],optimize="optimal") 
                                                         - 2 * np.einsum('ij,uv,tv->tiuj', id_cor, self.dm1_cas, self.F_core[ncore:nocc, ncore:nocc],optimize="optimal"))
            Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore] += ( 4 * np.einsum('ij,tu->tiuj', id_cor, F_tot[ncore:nocc, ncore:nocc],optimize="optimal") 
                                                         - 4 * np.einsum('tu,ij->tiuj', id_cas, F_tot[:ncore, :ncore],optimize="optimal"))

            #AM: I need to think about this
            Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore] = 0.5 * (Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore] + np.einsum('tiuj->ujti', Htmp[ncore:nocc,:ncore,ncore:nocc,:ncore],optimize="optimal"))

        return(Htmp)


    def get_hessianCICI(self):
        """ Build the CI-CI sector of the Hessian"""
        if(self.ndet > 1):
            e0 = self.energy
            return 2.0 * np.einsum('ki,kl,lj->ij', 
                    self.mat_ci[:,1:], self.ham - self.energy * np.identity(self.ndet), self.mat_ci[:,1:], optimize="optimal")
        else: 
            return np.zeros((0,0))

    
    def koopmans(self):
        """
        Solve IP using Koopmans theory
        """
        from scipy.linalg import eigh
        # Transform gen Fock matrix to MO basis
        gen_fock = self.gen_fock[:self.nocc,:self.nocc]
        # Compute the density matrix in occupied space
        gen_dens = np.zeros((self.nocc,self.nocc))
        gen_dens[:self.ncore,:self.ncore] = np.eye(self.ncore)*2
        gen_dens[self.ncore:,self.ncore:] = self.dm1_cas
        # Solve the generalized eigenvalue problem
        e, v = eigh(-gen_fock, gen_dens)
        # Normalize ionization orbitals wrt standard metric
        for i in range(self.nocc):
            v[:,i] /= np.linalg.norm(v[:,i])
        # Convert ionization orbitals to MO basis
        cip = self.mo_coeff[:,:self.nocc].dot(v)
        # Compute occupation of ionization orbitals
        occip = np.diag(np.einsum('ip,ij,jq->pq',v,gen_dens,v))   
        return e, cip, occip

    def get_civector(self):
        from pyscf.fci.cistring import gen_occslst
        
        occlst_a = gen_occslst(range(self.ncore,self.nocc),self.cas_nalfa)
        occlst_b = gen_occslst(range(self.ncore,self.nocc),self.cas_nbeta) 
        na = len(occlst_a)
        nb = len(occlst_b)
        core_list = list(range(0,self.ncore))

        for ia, occa in enumerate(occlst_a):
            for ib, occb in enumerate(occlst_b):
                yield self.mat_ci[ia*nb+ib,0], core_list+list(occa), core_list+list(occb)
        

    def canonicalize(self):
        """Canonicalise the natural orbitals and CI coefficients"""
        # TODO: Implement canonicalisation of SS-CASSCF state
        return
    
    def get_preconditioner(self):
        # TODO: Implement a preconditioner for approximate inverse Hessian
        return np.ones(self.dim)

    def uniq_var_indices(self, frozen):
        """ Create a mask indicating the non-redundant orbital rotations.
            Mask corresponds to a matrix of boolean with size (norb,norb).
        """
        mask = np.zeros((self.nmo,self.nmo),dtype=bool)
        # Active-Core rotations
        mask[self.ncore:self.nocc,:self.ncore] = True   
        # Virtual-Core and Virtual-Active rotations
        mask[self.nocc:,:self.nocc]            = True
        # Set any frozen orbitals to False
        if frozen is not None:
            if isinstance(frozen, (int, np.integer)):
                mask[:frozen] = mask[:,:frozen] = False
            else:
                frozen = np.asarray(frozen)
                mask[frozen] = mask[:,frozen] = False
        return mask
