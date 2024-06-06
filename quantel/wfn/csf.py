#!/usr/bin/python3
# Modified from ss_casscf code of Antoine Marie and Hugh G. A. Burton
# This is code for a CSF, which can be formed in a variety of ways.

import numpy as np
import scipy, quantel, h5py, warnings
from quantel.utils.csf_utils import get_csf_vector
from quantel.utils.linalg import delta_kron, orthogonalise
from quantel.gnme.csf_noci import csf_coupling
from .wavefunction import Wavefunction


class GenealogicalCSF(Wavefunction):
    """ 
        A class for a Configuration State Function (CSF) using the genealogical coupling pattern.

        Inherits from the Wavefunction abstract base class with pure virtual properties:
            - energy
            - gradient
            - hessian
            - take_step
            - save_last_step
            - restore_step
    """
    def __init__(self, integrals, spin_coupling, verbose=0, nohess=False):
        """ Initialise the CSF wave function
                integrals     : quantel integral interface
                spin_coupling : genealogical coupling pattern
                verbose       : verbosity level
        """
        self.spin_coupling = spin_coupling
        self.verbose       = verbose
        # Initialise integrals object
        self.integrals  = integrals
        self.nalfa      = integrals.molecule().nalfa()
        self.nbeta      = integrals.molecule().nbeta()
        # Initialise molecular integrals object
        self.mo_ints    = quantel.MOintegrals(integrals)
        # Get number of basis functions and linearly independent orbitals
        self.nbsf       = integrals.nbsf()
        self.nmo        = integrals.nmo()
        # Record whether Hessian allowed
        self.nohess     = nohess
    
    
    def sanity_check(self):
        '''Need to be run at the start of the kernel to verify that the number of 
           orbitals and electrons in the CAS are consistent with the system '''
        # Check number of active orbitals is positive
        if self.cas_nmo <= 0:
            raise ValueError("Number of active orbitals must be positive")
        # Check number of active electrons doesn't exceed total number of electrons
        if(self.cas_nalfa > self.cas_nmo or self.cas_nbeta > self.cas_nmo):
            raise ValueError("Number of active electrons must be <= number of active orbitals")
        # Check number of active electrons doesn't exceed total number of electrons
        if(self.cas_nalfa > self.nalfa or self.cas_nbeta > self.nbeta):
            raise ValueError("Number of active electrons must be <= total number of electrons")
        # Check number of occupied orbitals doesn't exceed total number of orbitals
        if(self.nocc > self.nmo):
            raise ValueError("Number of inactive and active orbitals must be <= total number of orbitals")
                             

    def initialise(self, mo_guess, spin_coupling=None, mat_ci=None, integrals=True):
        """ Initialise the CSF object with a set of MO coefficients"""
        if(spin_coupling is None):
            spin_coupling = self.spin_coupling
        
        # Save orbital coefficients
        mo_guess      = orthogonalise(mo_guess, self.integrals.overlap_matrix())
        if(mo_guess.shape[1] != self.nmo):
            raise ValueError("Number of orbitals in MO coefficient matrix is incorrect")
        self.mo_coeff = mo_guess

        # Get active space definition
        self.cas_nmo    = len(spin_coupling)
        self.cas_nalfa  = sum(int(s=='+') for s in spin_coupling)
        self.cas_nbeta  = sum(int(s=='-') for s in spin_coupling)
        # Get number of core electrons
        self.ncore = self.integrals.molecule().nelec() - self.cas_nalfa - self.cas_nbeta
        if(self.ncore % 2 != 0):
            raise ValueError("Number of core electrons must be even")
        if(self.ncore < 0):
            raise ValueError("Number of core electrons must be positive")
        self.ncore = self.ncore // 2
        # Get numer of 'occupied' orbitals
        self.nocc = self.ncore + self.cas_nmo
        self.sanity_check()

        # Get determinant list and coefficient vector
        self.spin_coupling = spin_coupling
        self.detlist, self.civec = get_csf_vector(spin_coupling)
        self.ndet = len(self.detlist)

        # Setup CI space
        self.cispace = quantel.CIspace(self.mo_ints, self.cas_nmo, self.cas_nalfa, self.cas_nbeta)
        self.cispace.initialize('custom', self.detlist)
        self.csf_dm1, self.csf_dm2 = self.get_active_rdm_12()

        # Save mapping indices for unique orbital rotations
        self.frozen     = None
        self.rot_idx    = self.uniq_var_indices(self.frozen)
        self.nrot       = np.sum(self.rot_idx)

        # Initialise integrals
        if (integrals): self.update_integrals()

    @property
    def dim(self):
        """Number of degrees of freedom"""
        return self.nrot

    @property
    def energy(self):
        """ Compute the energy corresponding to a given set of
             one-el integrals, two-el integrals, 1- and 2-RDM
        """
        E = self.energy_core
        E += np.einsum('pq,pq', self.h1eff, self.csf_dm1, optimize="optimal")
        E += 0.5 * np.einsum('pqrs,pqrs', self.h2eff, self.csf_dm2, optimize="optimal")
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
        rdm2ab = self.cispace.rdm2(self.civec,True,False).transpose(0,2,1,3)
        return abs(self.sz * (self.sz + 1)  + self.cas_nbeta - np.einsum('pqqp',rdm2ab))

    @property
    def gradient(self):
        """ Compute the gradient of the energy with respect to the orbital rotations"""
        return self.get_orbital_gradient()

    @property
    def hessian(self):
        ''' This method finds orb-orb part of the Hessian '''
        if(self.nohess):
            raise RuntimeError("Hessian calculation not allowed with 'nohess' flag")
        return (self.get_hessianOrbOrb()[:, :, self.rot_idx])[self.rot_idx, :]

    def get_active_rdm_12(self):
        """ Compute the 1- and 2-electron reduced density matrices in the active space.
            returns:
                dm1_csf: 1-electron reduced density matrix
                dm2_csf: 2-electron reduced density matrix
        """
        # Make RDM1
        csf_rdm1a = self.cispace.rdm1(self.civec, True)
        csf_rdm1b = self.cispace.rdm1(self.civec, False)
        csf_rdm1 = csf_rdm1a + csf_rdm1b
        # Make RDM2 (need to convert phys -> chem notation)
        csf_rdm2aa = self.cispace.rdm2(self.civec, True, True).transpose(0,2,1,3)
        csf_rdm2bb = self.cispace.rdm2(self.civec, False, False).transpose(0,2,1,3)
        csf_rdm2ab = self.cispace.rdm2(self.civec, True, False).transpose(0,2,1,3)
        csf_rdm2 = csf_rdm2aa + csf_rdm2bb + csf_rdm2ab + csf_rdm2ab.transpose(2,3,0,1)
        return csf_rdm1, csf_rdm2
    
    def update_integrals(self):
        """ Update the integrals with current set of orbital coefficients"""
        self.mo_ints.update_orbitals(self.mo_coeff,self.ncore,self.cas_nmo)
                
        # Effective integrals in active space
        self.energy_core = self.mo_ints.scalar_potential()
        self.h1eff = self.mo_ints.oei_matrix(True)
        self.h2eff = self.mo_ints.tei_array(True,False).transpose(0,2,1,3)

        # 1 and 2 electron integrals outside active space
        self.h1e = np.linalg.multi_dot([self.mo_coeff.T, self.integrals.oei_matrix(True), self.mo_coeff])
        Cocc = self.mo_coeff[:,:self.nocc].copy()
        self.pppo = self.integrals.tei_ao_to_mo(self.mo_coeff,self.mo_coeff,self.mo_coeff,Cocc,True,False).transpose(0,2,1,3)
        # Slices for easy indexing
        self.pooo = self.pppo[:,:self.nocc,:self.nocc,:self.nocc]
        self.ppoo = self.pppo[:,:,:self.nocc,:self.nocc]
        self.popo = self.pppo[:,:self.nocc,:,:self.nocc]

        #self.pooo = self.integrals.tei_ao_to_mo(self.mo_coeff,Cocc,Cocc,Cocc,True,False).transpose(0,2,1,3)
        #if(not self.nohess):
        #    self.ppoo = self.integrals.tei_ao_to_mo(self.mo_coeff,Cocc,self.mo_coeff,Cocc,True,False).transpose(0,2,1,3)
        #    self.popo = self.integrals.tei_ao_to_mo(self.mo_coeff,self.mo_coeff,Cocc,Cocc,True,False).transpose(0,2,1,3)

        # Construct core potential outside active space
        #dm_core = np.dot(self.mo_coeff[:,:self.ncore], self.mo_coeff[:,:self.ncore].T)
        #v_jk = self.integrals.build_JK(dm_core)
        #self.vhf_c = np.linalg.multi_dot([self.mo_coeff.T, v_jk, self.mo_coeff])
        self.vhf_c = (2 * np.einsum('pqii->pq',self.pppo[:,:,:self.ncore,:self.ncore],optimize='optimal')
                        - np.einsum('ipqi->pq',self.pppo[:self.ncore,:,:,:self.ncore],optimize='optimal'))

        # Fock matrices
        self.get_fock_matrices()
        return 

    
    def save_to_disk(self, tag):
        """Save a CSF to disk with prefix 'tag'"""
        # Save hdf5 file with mo coefficients and spin coupling
        with h5py.File(tag+'.hdf5','w') as F:
            F.create_dataset("mo_coeff", data=self.mo_coeff[:,:self.nocc])
            F.create_dataset("spin_coupling", data=self.spin_coupling)
            F.create_dataset("energy", data=self.energy)
            F.create_dataset("s2", data=self.s2)
        
        # Save numpy txt file with energy and Hessian index
        hindices = self.get_hessian_index()
        with open(f"{tag}.solution","w") as F:
            F.write(f"{self.energy:18.12f} {hindices[0]:5d} {hindices[1]:5d} {self.s2:12.6f}\n")
        return 
    
    def read_from_disk(self, tag):
        """Read a CSF wavefunction from disk with prefix 'tag'"""
        with h5py.File(tag+'.hdf5','r') as F:
            mo_read = F['mo_coeff'][:]
            spin_coupling = str(F['spin_coupling'][...])[2:-1]

        # Initialise the wave function
        self.initialise(mo_read, spin_coupling=spin_coupling)        
        
        # Check the input
        if mo_read.shape[0] != self.nbsf:
            raise ValueError("Inccorect number of AO basis functions in file")
        if mo_read.shape[1] < self.nocc:
            raise ValueError("Insufficient orbitals in file to represent occupied orbitals")
        if mo_read.shape[1] > self.nmo:
            raise ValueError("Too many orbitals in file")
        return

    def copy(self):
        """Return a copy of the current object"""
        newcsf = GenealogicalCSF(self.integrals, self.spin_coupling, verbose=self.verbose)
        newcsf.initialise(self.mo_coeff)
        return newcsf

    def overlap(self, them):
        """ Compute the overlap between two CSF objects
        """
        ovlp = self.integrals.overlap_matrix()
        return csf_coupling(self, them, ovlp)[0]

    def hamiltonian(self, them):
        """ Compute the Hamiltonian coupling between two CSF objects
        """
        hcore = self.integrals.oei_matrix(True)
        eri   = self.integrals.tei_array(True,False).transpose(0,2,1,3).reshape(self.nbsf**2,self.nbsf**2)
        ovlp  = self.integrals.overlap_matrix()
        enuc  = self.integrals.scalar_potential()
        return csf_coupling(self, them, ovlp, hcore, eri, enuc)

    def deallocate(self):
        """ Reduce the memory footprint for storing"""
        self.ppoo = None
        self.popo = None
        self.h1e = None
        self.h1eff = None
        self.h2eff = None
        self.F_core = None
        self.F_active = None

    def restore_last_step(self):
        """ Restore MO coefficients to previous step"""
        self.mo_coeff = self.mo_coeff_save.copy()
        self.update_integrals()

    def save_last_step(self):
        """ Save MO coefficients"""
        self.mo_coeff_save = self.mo_coeff.copy()

    def take_step(self, step):
        """ Take a step in the orbital space"""
        self.save_last_step()
        self.rotate_orb(step[:self.nrot])
        self.update_integrals()

    def rotate_orb(self, step):
        """ Rotate molecular orbital coefficients with a step"""
        orb_step = np.zeros((self.nmo, self.nmo))
        orb_step[self.rot_idx] = step
        self.mo_coeff = np.dot(self.mo_coeff, scipy.linalg.expm(orb_step - orb_step.T))

    def get_fock_matrices(self):
        ''' Compute the core part of the generalized Fock matrix '''
        # Core contribution is just effective 1-electron matrix
        self.F_core = self.h1e + self.vhf_c
        # Active space contribution
        Cactive = self.mo_coeff[:,self.ncore:self.nocc].copy()
        dm_active = np.linalg.multi_dot([Cactive, self.csf_dm1, Cactive.T])
        self.F_active = 0.5 * self.integrals.build_JK(dm_active)
        self.F_active = np.linalg.multi_dot([self.mo_coeff.T, self.F_active, self.mo_coeff])
        return

    def get_generalised_fock(self, csf_dm1, csf_dm2):
        """ Compute the generalised Fock matrix"""
        ncore = self.ncore
        nocc  = self.nocc
        F = np.zeros((self.nmo, self.nmo))
        F[:ncore, :] = 2 * (self.F_core[:,:ncore] + self.F_active[:, :ncore]).T
        F[ncore:nocc,:] += np.dot(csf_dm1, self.F_core[:,ncore:nocc].T)

        # 2-electron active space component
        F[ncore:nocc,:] += np.einsum('vwxy,nwxy->vn',csf_dm2,self.pooo[:,ncore:,ncore:,ncore:],optimize='optimal')

        return 2 * F.T

    def get_orbital_gradient(self):
        ''' This method builds the orbital part of the gradient '''
        g_orb = self.get_generalised_fock(self.csf_dm1, self.csf_dm2)
        return (g_orb - g_orb.T)[self.rot_idx]

    def get_hessianOrbOrb(self):
        ''' This method build the orb-orb part of the hessian '''
        norb = self.nmo
        ncore = self.ncore
        ncas = self.cas_nmo
        nocc = ncore + ncas
        nvir = norb - nocc

        Htmp = np.zeros((norb, norb, norb, norb))
        F_tot = self.F_core + self.F_active

        # Temporary identity matrices
        id_cor = np.identity(ncore)
        id_vir = np.identity(nvir)
        id_cas = np.identity(ncas)

        # virtual-core virtual-core H_{ai,bj}
        if ncore > 0 and nvir > 0:
            aibj = self.popo[nocc:,:ncore,nocc:,:ncore]
            abij = self.ppoo[nocc:,nocc:,:ncore,:ncore].transpose((0,2,1,3))

            Htmp[nocc:,:ncore,nocc:,:ncore] = 4*(4*aibj-abij-aibj.transpose((0,3,2,1)))
            for i in range(ncore):
                Htmp[nocc:,i,nocc:,i] += 4 * F_tot[nocc:,nocc:]
            for a in range(nocc,norb):
                Htmp[a,:ncore,a,:ncore] -= 4 * F_tot[:ncore,:ncore]

        # virtual-core virtual-active H_{ai,bt}
        if ncore > 0 and nvir > 0 and ncas > 0:
            aibv = self.popo[nocc:,:ncore,nocc:,ncore:nocc]
            avbi = self.popo[nocc:,ncore:nocc,nocc:,:ncore]
            abvi = self.ppoo[nocc:,nocc:,ncore:nocc,:ncore]

            Htmp[nocc:,:ncore,nocc:,ncore:nocc] = (
                2 * np.einsum('tv,aibv->aibt', self.csf_dm1,
                    4 * aibv - avbi.transpose((0,3,2,1))-abvi.transpose((0,3,1,2)),optimize="optimal")
                - 2 * np.einsum('ab,tvxy,vixy ->aibt', id_vir, 0.5 * self.csf_dm2,
                    self.ppoo[ncore:nocc,:ncore,ncore:nocc,ncore:nocc], optimize="optimal")
                - 2 * np.einsum('ab,ti->aibt',id_vir,F_tot[ncore:nocc,:ncore],optimize="optimal")
                - 1 * np.einsum('ab,tv,vi->aibt',id_vir,self.csf_dm1,self.F_core[ncore:nocc,:ncore],optimize="optimal"))

        # virtual-active virtual-core H_{bt,ai}
        if ncore > 0 and nvir > 0 and ncas > 0:
            Htmp[nocc:,ncore:nocc,nocc:,:ncore] = np.einsum(
                'aibt->btai',Htmp[nocc:,:ncore,nocc:,ncore:nocc],optimize="optimal")

        # virtual-core active-core H_{ai,tj}
        if ncore > 0 and nvir > 0 and ncas > 0:
            aivj = self.ppoo[nocc:,:ncore,ncore:nocc,:ncore]
            avji = self.ppoo[nocc:,ncore:nocc,:ncore,:ncore]
            ajvi = self.ppoo[nocc:,:ncore,ncore:nocc,:ncore]

            Htmp[nocc:,:ncore,ncore:nocc,:ncore] = (
                2 * np.einsum('tv,aivj->aitj', (2 * id_cas - self.csf_dm1),
                    4 * aivj - avji.transpose((0,3,1,2))-ajvi.transpose((0,3,2,1)),optimize="optimal")
                - np.einsum('ji,tvxy,avxy -> aitj',id_cor,self.csf_dm2,
                    self.ppoo[nocc:,ncore:nocc,ncore:nocc,ncore:nocc],optimize="optimal")
                + 4 * np.einsum('ij,at-> aitj',id_cor,F_tot[nocc:,ncore:nocc],optimize="optimal")
                - np.einsum('ij,tv,av-> aitj',id_cor,self.csf_dm1,self.F_core[nocc:,ncore:nocc],optimize="optimal"))

        # active-core virtual-core H_{tj,ai}
        if ncore > 0 and nvir > 0 and ncas > 0:
            Htmp[ncore:nocc, :ncore, nocc:, :ncore] = np.einsum('aitj->tjai', Htmp[nocc:, :ncore, ncore:nocc, :ncore],
                                                                optimize="optimal")

        # virtual-active virtual-active H_{at,bu}
        if nvir > 0 and ncas > 0:
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc] = (4 * np.einsum('tuvx,abvx->atbu', 0.5 * self.csf_dm2,
                                                                        self.ppoo[nocc:, nocc:, ncore:nocc, ncore:nocc],
                                                                        optimize="optimal")
                                                          + 4 * np.einsum('txvu,axbv->atbu', 0.5 * self.csf_dm2,
                                                                          self.popo[nocc:, ncore:nocc, nocc:,
                                                                          ncore:nocc], optimize="optimal")
                                                          + 4 * np.einsum('txuv,axbv->atbu', 0.5 * self.csf_dm2,
                                                                          self.popo[nocc:, ncore:nocc, nocc:,
                                                                          ncore:nocc], optimize="optimal"))
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc] -= (
                    2 * np.einsum('ab,tvxy,uvxy->atbu', id_vir, 0.5 * self.csf_dm2,
                                  self.ppoo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc], optimize="optimal")
                    + 1 * np.einsum('ab,tv,uv->atbu', id_vir, self.csf_dm1, self.F_core[ncore:nocc, ncore:nocc],
                                    optimize="optimal"))
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc] -= (
                    2 * np.einsum('ab,uvxy,tvxy->atbu', id_vir, 0.5 * self.csf_dm2,
                                  self.ppoo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc], optimize="optimal")
                    + 1 * np.einsum('ab,uv,tv->atbu', id_vir, self.csf_dm1, self.F_core[ncore:nocc, ncore:nocc],
                                    optimize="optimal"))
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc] += 2 * np.einsum('tu,ab->atbu', self.csf_dm1,
                                                                        self.F_core[nocc:, nocc:], optimize="optimal")

        # active-core virtual-active H_{ti,au}
        if ncore > 0 and nvir > 0 and ncas > 0:
            avti = self.ppoo[nocc:, ncore:nocc, ncore:nocc, :ncore]
            aitv = self.ppoo[nocc:, :ncore, ncore:nocc, ncore:nocc]

            Htmp[ncore:nocc, :ncore, nocc:, ncore:nocc] = (- 4 * np.einsum('tuvx,aivx->tiau', 0.5 * self.csf_dm2,
                                                                           self.ppoo[nocc:, :ncore, ncore:nocc,
                                                                           ncore:nocc], optimize="optimal")
                                                           - 4 * np.einsum('tvux,axvi->tiau', 0.5 * self.csf_dm2,
                                                                           self.ppoo[nocc:, ncore:nocc, ncore:nocc,
                                                                           :ncore], optimize="optimal")
                                                           - 4 * np.einsum('tvxu,axvi->tiau', 0.5 * self.csf_dm2,
                                                                           self.ppoo[nocc:, ncore:nocc, ncore:nocc,
                                                                           :ncore], optimize="optimal"))
            Htmp[ncore:nocc, :ncore, nocc:, ncore:nocc] += (2 * np.einsum('uv,avti->tiau', self.csf_dm1,
                                                                          4 * avti - aitv.transpose(
                                                                              (0, 3, 2, 1)) - avti.transpose(
                                                                              (0, 2, 1, 3)), optimize="optimal")
                                                            - 2 * np.einsum('tu,ai->tiau', self.csf_dm1,
                                                                            self.F_core[nocc:, :ncore],
                                                                            optimize="optimal")
                                                            + 2 * np.einsum('tu,ai->tiau', id_cas, F_tot[nocc:, :ncore],
                                                                            optimize="optimal"))

            # virtual-active active-core  H_{au,ti}
            Htmp[nocc:, ncore:nocc, ncore:nocc, :ncore] = np.einsum('auti->tiau',
                                                                    Htmp[ncore:nocc, :ncore, nocc:, ncore:nocc],
                                                                    optimize="optimal")

        # active-core active-core H_{ti,uj} Nick 18 Mar
        if ncore > 0 and ncas > 0:
            gixyj = self.popo[:ncore, ncore:nocc, ncore:nocc, :ncore]
            gtijx = self.popo[ncore:nocc, :ncore, :ncore, ncore:nocc]
            gtxji = self.popo[ncore:nocc, ncore:nocc, :ncore, :ncore]
            gtwxy = self.popo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc]
            tiuj = 2 * np.einsum("tu,ij->tiuj", self.csf_dm1, self.F_core[:ncore, :ncore]) + \
                   4 * np.einsum("ij,tu->tiuj", id_cor, F_tot[ncore:nocc, ncore:nocc]) - \
                   np.einsum("ij,tw,uw->tiuj", id_cor, self.csf_dm1, self.F_core[ncore:nocc, ncore:nocc]) - \
                   np.einsum("ij,twxy,uwxy->tiuj", id_cor, self.csf_dm2, gtwxy) - \
                   np.einsum("ij,uw,tw->tiuj", id_cor, self.csf_dm1, self.F_core[ncore:nocc, ncore:nocc]) - \
                   np.einsum("ij,uwxy,twxy->tiuj", id_cor, self.csf_dm2, gtwxy) - \
                   2 * np.einsum("tu,ji->tiuj", id_cas, F_tot[:ncore, :ncore]) - \
                   2 * np.einsum("tu,ij->tiuj", id_cas, F_tot[:ncore, :ncore]) + \
                   2 * np.einsum("xu,tijx->tiuj", id_cas - self.csf_dm1,
                                 4 * gtijx - gtijx.transpose((0, 2, 1, 3)) - gtxji.transpose((0, 3, 2, 1))) + \
                   2 * np.einsum("xt,xiju->tiuj", id_cas - self.csf_dm1,
                                 4 * gtijx - gtijx.transpose((0, 2, 1, 3)) - gtxji.transpose((0, 3, 2, 1))) + \
                   2 * np.einsum("txuy,ixyj->tiuj", self.csf_dm2, gixyj) + \
                   2 * np.einsum("txyu,ixyj->tiuj", self.csf_dm2, gixyj) + \
                   2 * np.einsum("tuxy,ijxy->tiuj", self.csf_dm2, gtxji.transpose((3, 2, 1, 0)))
            Htmp[ncore:nocc, :ncore, ncore:nocc, :ncore] = tiuj
            Htmp[ncore:nocc, :ncore, ncore:nocc, :ncore] = np.einsum("tiuj->ujti", tiuj)

        # Nick: Active-active Hessian contributions
        # active-active active-core H_{xy,ti}
        if ncore > 0 and ncas > 0:
            gxvit = self.ppoo[ncore:nocc, ncore:nocc, :ncore, ncore:nocc]
            gxivt = self.popo[ncore:nocc, :ncore, ncore:nocc, ncore:nocc]
            gxtiv = self.ppoo[ncore:nocc, ncore:nocc, :ncore, ncore:nocc]
            xyti = 2 * np.einsum("xt,yi->xyti", self.csf_dm1, self.F_core[ncore:nocc, :ncore], optimize="optimal") + \
                   2 * np.einsum("xvtw,yvwi->xyti", self.csf_dm2, self.popo[ncore:nocc, ncore:nocc, ncore:nocc, :ncore],
                                 optimize="optimal") + \
                   2 * np.einsum("xvwt,yvwi->xyti", self.csf_dm2, self.popo[ncore:nocc, ncore:nocc, ncore:nocc, :ncore],
                                 optimize="optimal") + \
                   2 * np.einsum("xtvw,yivw->xyti", self.csf_dm2, self.popo[ncore:nocc, :ncore, ncore:nocc, ncore:nocc],
                                 optimize="optimal") + \
                   2 * np.einsum("yv,xvit->xyti", self.csf_dm1,
                                 4 * gxvit - gxivt.transpose((0, 2, 1, 3)) - gxtiv.transpose((0, 3, 2, 1)),
                                 optimize="optimal") + \
                   np.einsum("yt,xw,iw->xyti", id_cas, self.csf_dm1, self.F_core[:ncore, ncore:nocc],
                             optimize="optimal") + \
                   np.einsum("yt,xuwz,iuwz->xyti", id_cas, self.csf_dm2,
                             self.popo[:ncore, ncore:nocc, ncore:nocc, ncore:nocc],
                             optimize="optimal") + \
                   2 * np.einsum("xi,yt->xyti", F_tot[ncore:nocc, :ncore], id_cas, optimize="optimal")
            Htmp[ncore:nocc, ncore:nocc, ncore:nocc, :ncore] = xyti - np.einsum("xyti->yxti", xyti)
        # active-core active-active H_{ti, xy}
        if ncore > 0 and ncas > 0:
            Htmp[ncore:nocc, :ncore, ncore:nocc, ncore:nocc] = np.einsum("xyti->tixy",
                                                                         Htmp[ncore:nocc, ncore:nocc, ncore:nocc,
                                                                         :ncore])

        # active-active virtual-core H_{xy,ai}, as well as virtual-core active-active H_{ai,xy}
        if ncore > 0 and nvir > 0 and ncas > 0:
            gyvai = self.popo[ncore:nocc, ncore:nocc, nocc:, :ncore]
            gyiav = self.popo[ncore:nocc, :ncore, nocc:, ncore:nocc]
            gayiv = self.popo[nocc:, ncore:nocc, :ncore, ncore:nocc]
            Yxyia = 2 * np.einsum("xv,yvai->xyai", self.csf_dm1,
                                  4 * gyvai - gyiav.transpose((0, 3, 2, 1)) - gayiv.transpose((1, 3, 0, 2)),
                                  optimize="optimal")
            Htmp[ncore:nocc, ncore:nocc, nocc:, :ncore] = -Yxyia + np.einsum("xyai->yxai", Yxyia)
            Htmp[nocc:, :ncore, ncore:nocc, ncore:nocc] = np.einsum("xyai->aixy",
                                                                    Htmp[ncore:nocc, ncore:nocc, nocc:, :ncore])

        # active-active virtual-active H_{xy,at}
        if nvir > 0 and ncas > 0:
            xyat = 2 * np.einsum("yt,xa->xyat", self.csf_dm1, self.F_core[ncore:nocc, nocc:], optimize="optimal") + \
                   np.einsum("xt,aw,yw->xyat", id_cas, self.F_core[nocc:, ncore:nocc], self.csf_dm1,
                             optimize="optimal") + \
                   np.einsum("xt,yuwz,auwz->xyat", id_cas, self.csf_dm2,
                             self.popo[nocc:, ncore:nocc, ncore:nocc, ncore:nocc], optimize="optimal") + \
                   2 * np.einsum("yvtw,xvaw->xyat", self.csf_dm2, self.popo[ncore:nocc, ncore:nocc, nocc:, ncore:nocc],
                                 optimize="optimal") + \
                   2 * np.einsum("yvwt,xvaw->xyat", self.csf_dm2, self.popo[ncore:nocc, ncore:nocc, nocc:, ncore:nocc],
                                 optimize="optimal") + \
                   2 * np.einsum("ytvw,axvw->xyat", self.csf_dm2, self.popo[nocc:, ncore:nocc, ncore:nocc, ncore:nocc],
                                 optimize="optimal")
            Htmp[ncore:nocc, ncore:nocc, nocc:, ncore:nocc] = xyat - np.einsum("xyat->yxat", xyat)

        # virtual-active active-active H_{at, xy}
        if nvir > 0 and ncas > 0:
            Htmp[nocc:, ncore:nocc, ncore:nocc, ncore:nocc] = np.einsum("xyat->atxy",
                                                                        Htmp[ncore:nocc, ncore:nocc, nocc:, ncore:nocc])

        # active-active active-active H_{xy,tv}
        if ncas > 0:
            xytv = 2 * np.einsum("xt,yv->xytv", self.csf_dm1, self.F_core[ncore:nocc, ncore:nocc], optimize="optimal") + \
                   2 * np.einsum("xwtz,ywzv->xytv", self.csf_dm2,
                                 self.popo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc],
                                 optimize="optimal") + \
                   2 * np.einsum("xwzt,ywzv->xytv", self.csf_dm2,
                                 self.popo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc],
                                 optimize="optimal") + \
                   2 * np.einsum("xtwz,yvwz->xytv", self.csf_dm2,
                                 self.popo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc],
                                 optimize="optimal") - \
                   np.einsum("yv,xw,tw->xytv", id_cas, self.csf_dm1, self.F_core[ncore:nocc, ncore:nocc],
                             optimize="optimal") - \
                   np.einsum("yv,tw,xw->xytv", id_cas, self.csf_dm1, self.F_core[ncore:nocc, ncore:nocc],
                             optimize="optimal") - \
                   np.einsum("yv,xuwz,tuwz->xytv", id_cas, self.csf_dm2,
                             self.popo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc],
                             optimize="optimal") - \
                   np.einsum("yv,tuwz,xuwz->xytv", id_cas, self.csf_dm2,
                             self.popo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc],
                             optimize="optimal")
            Htmp[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc] = xytv - \
                                                                   np.einsum("xytv->yxtv", xytv) - \
                                                                   np.einsum("xytv->xyvt", xytv) + \
                                                                   np.einsum("xytv->yxvt", xytv)
        return (Htmp)

    def edit_mask_by_gcoupling(self, mask):
        r"""
        This function looks at the genealogical coupling scheme and modifies a given mask.
        The mask restricts the number of free parameters.

        The algorithm works by looking at each column and traversing downwards the columns.
        """
        g_coupling_arr = list(self.spin_coupling)
        n_dim = len(g_coupling_arr)
        for i, gfunc in enumerate(g_coupling_arr):  # This is for the columns
            for j in range(i + 1, n_dim):  # This is for the rows
                if gfunc == g_coupling_arr[j]:
                    mask[j, i] = False
                else:
                    break
        return mask

    def uniq_var_indices(self, frozen):
        """ This function creates a matrix of boolean of size (norb,norb).
            A True element means that this rotation should be taken into
            account during the optimization. Taken from pySCF.mcscf.casscf
        """
        mask = np.zeros((self.nmo, self.nmo), dtype=bool)
        # Active-core rotations
        mask[self.ncore:self.nocc, :self.ncore] = True
        # Virtual-Core and Virtual-Active rotations
        mask[self.nocc:, :self.nocc] = True
        # Active-Active rotations
        mask[self.ncore:self.nocc, self.ncore:self.nocc] = np.tril(
            np.ones((self.cas_nmo, self.cas_nmo), dtype=bool), k=-1)
        
        # Modify for genealogical coupling
        if self.spin_coupling is not None:
            mask[self.ncore:self.nocc, self.ncore:self.nocc] = self.edit_mask_by_gcoupling(
                mask[self.ncore:self.nocc,self.ncore:self.nocc])

        # Account for any frozen orbitals   
        if frozen is not None:
            if isinstance(frozen, (int, np.integer)):
                mask[:frozen] = mask[:, :frozen] = False
            else:
                frozen = np.asarray(frozen)
                mask[frozen] = mask[:, frozen] = False
        return mask

    def get_gcoupling_partition(self):
        r"""
        Partition a genealogical coupling scheme.
        e.g. ++-- -> [2, 2]
             +-+++--+ -> [1, 1, 3, 2, 1]
        For ease of use, we will convert the integer array (A) into another array of equal dimension (B),
        such that B[n] = A[0] + A[1] + ... +  A[n-1]
        """
        arr = []
        g_coupling_arr = list(self.g_coupling)
        count = 1
        ref = g_coupling_arr[0]
        for i, gfunc in enumerate(g_coupling_arr[1:]):
            if gfunc == ref:
                count += 1
            else:
                arr.append(count)
                ref = gfunc
                count = 1
        remainder = len(g_coupling_arr) - np.sum(arr)
        arr.append(remainder)
        partition_instructions = [0]
        for i, dim in enumerate(arr):
            partition_instructions.append(dim + partition_instructions[-1])
        return partition_instructions

    def canonicalize(self):
        """
        Forms the canonicalised MO coefficients by diagonalising invariant subblocks of the Fock matrix
        """
        fock = self.F_core + self.F_active
        # Get occ-occ and vir-vir blocks of (pseudo) Fock matrix
        foo = fock[:self.ncore, :self.ncore]
        faa = fock[self.ncore:self.nocc, self.ncore:self.nocc]
        fvv = fock[self.nocc:, self.nocc:]
        # Get transformations
        self.mo_energy = np.zeros(self.nmo)
        self.mo_energy[:self.ncore], Qoo = np.linalg.eigh(foo)
        self.mo_energy[self.nocc:], Qvv = np.linalg.eigh(fvv)
        self.mo_energy[self.ncore:self.nocc] = np.diag(faa)
        # Apply transformations
        self.mo_coeff[:,:self.ncore] = np.dot(self.mo_coeff[:,:self.ncore], Qoo)
        self.mo_coeff[:,self.nocc:] = np.dot(self.mo_coeff[:,self.nocc:], Qvv)
        
        # Update integrals
        self.update_integrals()

        # Set occupation numbers
        self.mo_occ = np.zeros(self.nmo)
        self.mo_occ[:self.ncore] = 2
        self.mo_occ[self.ncore:self.nocc] = 1


        return
