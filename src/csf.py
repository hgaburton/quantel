#!/usr/bin/python3
# Modified from ss_casscf code of Antoine Marie and Hugh G. A. Burton
# This is code for a single CSF case -- just to understand what is going on here

import numpy as np
import scipy
from functools import reduce
from typing import List
from pyscf import scf, fci, __config__, ao2mo, lib, mcscf
from csfs.ConfigurationStateFunctions.CSF import ConfigurationStateFunction
from csfs.Operators.Operators import get_generic_no_overlap
from utils import delta_kron, orthogonalise


class csf():
    def __init__(self, mol, s, ncas, nelecas, frozen: int, core: List[int], act: List[int], g_coupling: str = None,
                 permutation: List[int] = None, mo_basis: 'str' = 'site', ncore=None):
        self.mol = mol
        self.s = s    # Value of S
        self.core = core
        self.act = act
        self.g_coupling = g_coupling
        self.permutation = permutation
        self.mo_basis = mo_basis
        self.nelec = mol.nelec
        self._scf = scf.RHF(mol)
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.max_memory = self._scf.max_memory
        self.ncas = ncas  # Number of active orbitals
        if isinstance(nelecas, (int, np.integer)):
            nelecb = (nelecas - mol.spin) // 2
            neleca = nelecas - nelecb
            self.nelecas = (neleca, nelecb)  # Tuple of number of active electrons
        else:
            self.nelecas = np.asarray((nelecas[0], nelecas[1])).astype(int)

        if ncore is None:
            ncorelec = self.mol.nelectron - sum(self.nelecas)
            assert ncorelec % 2 == 0
            assert ncorelec >= 0
            self.ncore = ncorelec // 2
        else:
            self.ncore = ncore
        # Get AO integrals
        self.get_ao_integrals()

        # Get information of CSFs
        self.csf_info = ConfigurationStateFunction(self.mol, self.s, self.core, self.act,
                                                   self.g_coupling, self.permutation, mo_basis=self.mo_basis)

        # Get number of determinants (which is the dimension of the problem)
        self.nDet = self.csf_info.n_dets

        # Save mapping indices for unique orbital rotations
        self.frozen = frozen
        self.rot_idx = self.uniq_var_indices(self.norb, self.frozen)
        self.nrot = np.sum(self.rot_idx)

        # Dimensions of problem
        self.dim = self.nrot

    def copy(self):
        # Return a copy of the current object
        newcsf = csf(self.mol, self.s, self.ncas, self.nelecas, self.frozen, self.core, self.act, self.g_coupling, self.permutation, self.mo_basis)
        newcsf.initialise(self.mo_coeff, integrals=False)
        return newcsf

    def overlap(self, ref):
        r"""
        Determines the overlap between the current CSF and a reference CSF.

        :param ref: A ConfigurationStateFunction object which we are comparing to
        """
        csf_coeffs = self.csf_info.csf_coeffs
        ref_coeffs = ref.csf_info.csf_coeffs
        smo = np.einsum("ip,ij,jq->pq", self.mo_coeff, self.ovlp, ref.mo_coeff)
        cross_overlap_mat = scipy.linalg.block_diag(smo, smo)
        return get_generic_no_overlap(self.csf_info.dets_sq, ref.csf_info.dets_sq, csf_coeffs, ref_coeffs,
                                      cross_overlap_mat)

    def sanity_check(self):
        '''Need to be run at the start of the kernel to verify that the number of
           orbitals and electrons in the CAS are consistent with the system '''
        assert self.ncas > 0
        ncore = self.ncore
        nvir = self.mo_coeff.shape[1] - ncore - self.ncas
        assert ncore >= 0
        assert nvir >= 0
        assert ncore * 2 + sum(self.nelecas) == self.mol.nelectron
        assert 0 <= self.nelecas[0] <= self.ncas
        assert 0 <= self.nelecas[1] <= self.ncas
        return self

    def get_ao_integrals(self):
        self.enuc = self._scf.energy_nuc()
        self.v1e = self.mol.intor('int1e_nuc')  # Nuclear repulsion matrix elements
        self.t1e = self.mol.intor('int1e_kin')  # Kinetic energy matrix elements
        self.hcore = self.t1e + self.v1e  # 1-electron matrix elements in the AO basis
        self.norb = self.hcore.shape[0]
        self.ovlp = self.mol.intor('int1e_ovlp')  # Overlap matrix
        self._scf._eri = self.mol.intor("int2e", aosym="s8")  # Two electron integrals

    def initialise(self, mo_guess=None, integrals=True):
        # Save orbital coefficients
        if(not(mo_guess is not None)): mo_guess = self.csf_info.coeffs.copy()
        mo_guess = orthogonalise(mo_guess, self.ovlp)
        self.mo_coeff = mo_guess.copy()
        self.nmo = self.mo_coeff.shape[1]

        # Initialise integrals
        if (integrals): self.update_integrals()

    def deallocate(self):
        # Reduce the memory footprint for storing
        self._eri = None
        self.ppoo = None
        self.popo = None
        self.h1e = None
        self.h1eff = None
        self.h2eff = None
        self.F_core = None
        self.F_cas = None

    @property
    def energy(self):
        ''' Compute the energy corresponding to a given set of
             one-el integrals, two-el integrals, 1- and 2-RDM '''
        E = self.energy_core
        Ecore = E
        E += np.einsum('pq,pq', self.h1eff, self.dm1_cas, optimize="optimal")
        onee = E - Ecore
        E += 0.5 * np.einsum('pqrs,pqrs', self.h2eff, self.dm2_cas, optimize="optimal")
        return E

    @property
    def s2(self):
        ''' Compute the spin of a given FCI vector '''
        return self.csf_info.get_s2()

    @property
    def gradient(self):
        r"""
        Unlike in CAS case, we are only interested in orbital optimisation
        :return:
        """
        g_orb = self.get_orbital_gradient()
        return g_orb

    @property
    def hessian(self):
        ''' This method finds orb-orb part of the Hessian '''
        H_OrbOrb = (self.get_hessianOrbOrb()[:, :, self.rot_idx])[self.rot_idx, :]
        return H_OrbOrb

    def get_hessian_index(self, tol=1e-16):
        eigs = scipy.linalg.eigvalsh(self.hessian)
        ndown = 0
        nzero = 0
        nuphl = 0
        for i in eigs:
            if i < -tol:
                ndown += 1
            elif i > tol:
                nuphl += 1
            else:
                nzero += 1
        return ndown, nzero, nuphl

    def pushoff(self, n, angle=np.pi / 2):
        """Perturb along n Hessian directions"""
        eigval, eigvec = np.linalg.eigh(self.hessian)
        step = sum(eigvec[:, i] * angle for i in range(n))
        self.take_step(step)

    def update_integrals(self):
        # One-electron Hamiltonian
        self.h1e = np.einsum('ip,ij,jq->pq', self.mo_coeff, self.hcore, self.mo_coeff, optimize="optimal")

        # Occupied orbitals
        nocc = self.ncore + self.ncas
        Cocc = self.mo_coeff[:, :nocc]
        self.ppoo = ao2mo.incore.general(self._scf._eri, (Cocc, Cocc, self.mo_coeff, self.mo_coeff), compact=False)
        self.ppoo = self.ppoo.reshape((nocc, nocc, self.nmo, self.nmo)).transpose(2, 3, 0, 1)
        self.popo = ao2mo.incore.general(self._scf._eri, (Cocc, self.mo_coeff, Cocc, self.mo_coeff), compact=False)
        self.popo = self.popo.reshape((nocc, self.nmo, nocc, self.nmo)).transpose(1, 0, 3, 2)

        # Get core potential
        mo_core = self.mo_coeff[:, :self.ncore]
        dm_core = np.dot(mo_core, mo_core.T)
        vj, vk = self._scf.get_jk(self.mol, dm_core)
        self.vhf_c = reduce(np.dot, (self.mo_coeff.T, 2 * vj - vk, self.mo_coeff))

        # Effective Hamiltonians in CAS space
        self.h1eff, self.energy_core = self.get_h1eff()
        self.h2eff = self.get_h2eff()
        self.h2eff = ao2mo.restore(1, self.h2eff, self.ncas)
        self.csf_info.update_coeffs(self.mo_coeff)
        # Reduced density matrices
        self.dm1_cas, self.dm2_cas = self.get_csfrdm_12(self.csf_info)
        # Transform 1e integrals
        self.h1e_mo = reduce(np.dot, (self.mo_coeff.T, self.hcore, self.mo_coeff))

        # Fock matrices
        self.get_fock_matrices()

    def restore_last_step(self):
        # Restore coefficients
        self.mo_coeff = self.mo_coeff_save.copy()

        # Finally, update our integrals for the new coefficients
        self.update_integrals()

    def save_last_step(self):
        self.mo_coeff_save = self.mo_coeff.copy()

    def take_step(self, step):
        # Save our last position
        self.save_last_step()

        # Take steps in orbital and CI space
        self.rotate_orb(step[:self.nrot])

        # Finally, update our integrals for the new coefficients
        self.update_integrals()

    def rotate_orb(self, step):
        orb_step = np.zeros((self.norb, self.norb))
        orb_step[self.rot_idx] = step
        self.mo_coeff = np.dot(self.mo_coeff, scipy.linalg.expm(orb_step - orb_step.T))

    def get_h1eff(self):
        '''CAS space one-electron hamiltonian

        Returns:
            A tuple, the first is the effective one-electron hamiltonian defined in CAS space,
            the second is the electronic energy from core.
        '''
        ncas = self.ncas
        ncore = self.ncore
        nocc = self.ncore + self.ncas

        # Get core and active orbital coefficients
        mo_core = self.mo_coeff[:, :ncore]
        mo_cas = self.mo_coeff[:, ncore:ncore + ncas]

        # Core density matrix (in AO basis)
        self.core_dm = np.dot(mo_core, mo_core.T) * 2

        # Core energy
        energy_core = self.enuc
        energy_core += np.einsum('ij,ji', self.core_dm, self.hcore, optimize="optimal")
        energy_core += self.vhf_c[:ncore, :ncore].trace()

        # Get effective Hamiltonian in CAS space
        h1eff = np.einsum('ki,kl,lj->ij', mo_cas.conj(), self.hcore, mo_cas, optimize="optimal")
        h1eff += self.vhf_c[ncore:nocc, ncore:nocc]
        return h1eff, energy_core

    def get_h2eff(self):
        '''Compute the active space two-particle Hamiltonian. '''
        nocc = self.ncore + self.ncas
        ncore = self.ncore
        return self.ppoo[ncore:nocc, ncore:nocc, ncore:, ncore:]

    def get_csfrdm_12(self, csfobj):
        r"""
        We first form a new CSFConstructor object with new coeffs
        :return:
        """
        # Nick 2 March 2023: Commented below two lines. Using PySCF to construct RDMs instead
        #dm1_csf = csfobj.get_csf_one_rdm()
        #dm2_csf = csfobj.get_csf_two_rdm()
        dm1_csf, dm2_csf = csfobj.get_pyscf_rdms()
        # Since we only want the active orbitals:
        #nocc = self.ncore + self.ncas
        #return dm1_csf[self.ncore:nocc, :][:, self.ncore:nocc],\
        #        dm2_csf[self.ncore:nocc, :, :, :][:, self.ncore:nocc, :, :][:, :, self.ncore:nocc, :][:, :, :, self.ncore:nocc]
        return dm1_csf, dm2_csf

    def get_fock_matrices(self):
        ''' Compute the core part of the generalized Fock matrix '''
        ncore = self.ncore

        # Full Fock
        vj = np.empty((self.nmo, self.nmo))
        vk = np.empty((self.nmo, self.nmo))
        for i in range(self.nmo):
            #print("RDM1 shape: ", self.dm1_cas.shape)
            #print("ppoo contracted shape: ", self.ppoo[i, :, ncore:, ncore:].shape)
            #print("ppoo shape: ", self.ppoo.shape)
            vj[i] = np.einsum('ij,qij->q', self.dm1_cas, self.ppoo[i, :, ncore:, ncore:], optimize="optimal")
            vk[i] = np.einsum('ij,iqj->q', self.dm1_cas, self.popo[i, ncore:, :, ncore:], optimize="optimal")
        fock = self.h1e_mo + self.vhf_c + vj - vk * 0.5

        # Core contribution
        self.F_core = self.h1e + self.vhf_c

        # Active space contribution
        self.F_cas = fock - self.F_core

        return

    def get_gen_fock(self, dm1_cas, dm2_cas, transition=False):
        """Build generalised Fock matrix"""
        ncore = self.ncore
        nocc = self.ncore + self.ncas

        # Effective Coulomb and exchange operators for active space
        J_a = np.empty((self.nmo, self.nmo))
        K_a = np.empty((self.nmo, self.nmo))
        for i in range(self.nmo):
            J_a[i] = np.einsum('ij,qij->q', dm1_cas, self.ppoo[i, :, ncore:, ncore:], optimize="optimal")
            K_a[i] = np.einsum('ij,iqj->q', dm1_cas, self.popo[i, ncore:, :, ncore:], optimize="optimal")
        V_a = 2 * J_a - K_a

        # Universal contributions
        tdm1_cas = dm1_cas + dm1_cas.T
        F = np.zeros((self.nmo, self.nmo))
        if (not transition):
            F[:ncore, :ncore] = 4.0 * self.F_core[:ncore, :ncore]
            F[ncore:, :ncore] = 4.0 * self.F_core[ncore:, :ncore]
        F[:, ncore:nocc] += np.einsum('qx,yq->xy', self.F_core[ncore:nocc, :], tdm1_cas, optimize="optimal")

        # Core-Core specific terms
        F[:ncore, :ncore] += V_a[:ncore, :ncore] + V_a[:ncore, :ncore].T

        # Active-Active specific terms (Nick modified)
        tdm2_cas = dm2_cas + dm2_cas.transpose(1, 0, 2, 3)
        Ftmp = np.empty((self.ncas, self.ncas))
        for i in range(self.ncas):
            Ftmp[i, :] = np.einsum('qrs,yqrs->y', self.ppoo[ncore + i, ncore:nocc, ncore:, ncore:], tdm2_cas,
                                   optimize="optimal")
        Ftmp += np.einsum('xw,vw->vx', self.F_core[ncore:nocc, ncore:nocc], tdm1_cas, optimize="optimal")
        F[ncore:nocc, ncore:nocc] = Ftmp

        # Core-Active specific terms
        F[ncore:nocc, :ncore] += V_a[ncore:nocc, :ncore] + (V_a[:ncore, ncore:nocc]).T

        # Effective interaction with orbitals outside active space
        ext_int = np.empty((self.nmo, self.ncas))
        for i in range(self.nmo):
            ext_int[i, :] = np.einsum('prs,pars->a', self.popo[i, ncore:, ncore:nocc, ncore:], tdm2_cas,
                                      optimize="optimal")

        # Active-core
        F[:ncore, ncore:nocc] += ext_int[:ncore, :]
        # Active-virtual
        F[nocc:, ncore:nocc] += ext_int[nocc:, :]

        # Core-virtual
        F[nocc:, :ncore] += V_a[nocc:, :ncore] + (V_a[:ncore, nocc:]).T
        return F

    def get_generalised_fock(self, dm1_cas, dm2_cas):
        ''' This method finds the generalised Fock matrix with a different method '''
        ncore = self.ncore
        nocc = self.ncore + self.ncas
        F = np.zeros((self.nmo, self.nmo))
        F[:ncore, :] = 2 * (self.F_core[:, :ncore] + self.F_cas[:, :ncore]).T
        F[ncore:nocc, :] = np.einsum("nw,vw->vn", self.F_core[:, ncore:nocc], dm1_cas) + \
                np.einsum("vwxy,nwxy->vn", dm2_cas, self.popo[:, ncore:nocc, ncore:nocc, ncore:nocc])
        return 2 * F.T


    def get_orbital_gradient(self):
        ''' This method builds the orbital part of the gradient '''
        #g_orb = self.get_gen_fock(self.dm1_cas, self.dm2_cas, False)
        g_orb = self.get_generalised_fock(self.dm1_cas, self.dm2_cas)
        return (g_orb - g_orb.T)[self.rot_idx]

    def get_hessianOrbOrb(self):
        ''' This method build the orb-orb part of the hessian '''
        norb = self.norb
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        nvir = norb - nocc

        Htmp = np.zeros((norb, norb, norb, norb))
        F_tot = self.F_core + self.F_cas

        # Temporary identity matrices
        id_cor = np.identity(ncore)
        id_vir = np.identity(nvir)
        id_cas = np.identity(ncas)

        # virtual-core virtual-core H_{ai,bj}
        if ncore > 0 and nvir > 0:
            aibj = self.popo[nocc:, :ncore, nocc:, :ncore]
            abij = self.ppoo[nocc:, nocc:, :ncore, :ncore]

            Htmp[nocc:, :ncore, nocc:, :ncore] = (
                    4 * (4 * aibj - abij.transpose((0, 2, 1, 3)) - aibj.transpose((0, 3, 2, 1)))
                    + 4 * np.einsum('ij,ab->aibj', id_cor, F_tot[nocc:, nocc:], optimize="optimal")
                    - 4 * np.einsum('ab,ij->aibj', id_vir, F_tot[:ncore, :ncore], optimize="optimal"))

        # virtual-core virtual-active H_{ai,bt}
        if ncore > 0 and nvir > 0:
            aibv = self.popo[nocc:, :ncore, nocc:, ncore:nocc]
            avbi = self.popo[nocc:, ncore:nocc, nocc:, :ncore]
            abvi = self.ppoo[nocc:, nocc:, ncore:nocc, :ncore]

            Htmp[nocc:, :ncore, nocc:, ncore:nocc] = (2 * np.einsum('tv,aibv->aibt', self.dm1_cas,
                                                                    4 * aibv - avbi.transpose(
                                                                        (0, 3, 2, 1)) - abvi.transpose((0, 3, 1, 2)),
                                                                    optimize="optimal")
                                                      - 2 * np.einsum('ab,tvxy,vixy ->aibt', id_vir, 0.5 * self.dm2_cas,
                                                                      self.ppoo[ncore:nocc, :ncore, ncore:nocc,
                                                                      ncore:nocc], optimize="optimal")
                                                      - 2 * np.einsum('ab,ti->aibt', id_vir, F_tot[ncore:nocc, :ncore],
                                                                      optimize="optimal")
                                                      - 1 * np.einsum('ab,tv,vi->aibt', id_vir, self.dm1_cas,
                                                                      self.F_core[ncore:nocc, :ncore],
                                                                      optimize="optimal"))

        # virtual-active virtual-core H_{bt,ai}
        if ncore > 0 and nvir > 0:
            Htmp[nocc:, ncore:nocc, nocc:, :ncore] = np.einsum('aibt->btai', Htmp[nocc:, :ncore, nocc:, ncore:nocc],
                                                               optimize="optimal")

        # virtual-core active-core H_{ai,tj}
        if ncore > 0 and nvir > 0:
            aivj = self.ppoo[nocc:, :ncore, ncore:nocc, :ncore]
            avji = self.ppoo[nocc:, ncore:nocc, :ncore, :ncore]
            ajvi = self.ppoo[nocc:, :ncore, ncore:nocc, :ncore]

            Htmp[nocc:, :ncore, ncore:nocc, :ncore] = (2 * np.einsum('tv,aivj->aitj', (2 * id_cas - self.dm1_cas),
                                                                     4 * aivj - avji.transpose(
                                                                         (0, 3, 1, 2)) - ajvi.transpose((0, 3, 2, 1)),
                                                                     optimize="optimal")
                                                       - np.einsum('ji,tvxy,avxy -> aitj', id_cor,
                                                                       self.dm2_cas,
                                                                       self.ppoo[nocc:, ncore:nocc, ncore:nocc,
                                                                       ncore:nocc], optimize="optimal")
                                                       + 4 * np.einsum('ij,at-> aitj', id_cor, F_tot[nocc:, ncore:nocc],
                                                                       optimize="optimal")
                                                       - np.einsum('ij,tv,av-> aitj', id_cor, self.dm1_cas,
                                                                       self.F_core[nocc:, ncore:nocc],
                                                                       optimize="optimal"))

        # active-core virtual-core H_{tj,ai}
        if ncore > 0 and nvir > 0:
            Htmp[ncore:nocc, :ncore, nocc:, :ncore] = np.einsum('aitj->tjai', Htmp[nocc:, :ncore, ncore:nocc, :ncore],
                                                                optimize="optimal")

        # virtual-active virtual-active H_{at,bu}
        if nvir > 0:
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc] = (4 * np.einsum('tuvx,abvx->atbu', 0.5 * self.dm2_cas,
                                                                        self.ppoo[nocc:, nocc:, ncore:nocc, ncore:nocc],
                                                                        optimize="optimal")
                                                          + 4 * np.einsum('txvu,axbv->atbu', 0.5 * self.dm2_cas,
                                                                          self.popo[nocc:, ncore:nocc, nocc:,
                                                                          ncore:nocc], optimize="optimal")
                                                          + 4 * np.einsum('txuv,axbv->atbu', 0.5 * self.dm2_cas,
                                                                          self.popo[nocc:, ncore:nocc, nocc:,
                                                                          ncore:nocc], optimize="optimal"))
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc] -= (
                    2 * np.einsum('ab,tvxy,uvxy->atbu', id_vir, 0.5 * self.dm2_cas,
                                  self.ppoo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc], optimize="optimal")
                    + 1 * np.einsum('ab,tv,uv->atbu', id_vir, self.dm1_cas, self.F_core[ncore:nocc, ncore:nocc],
                                    optimize="optimal"))
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc] -= (
                    2 * np.einsum('ab,uvxy,tvxy->atbu', id_vir, 0.5 * self.dm2_cas,
                                  self.ppoo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc], optimize="optimal")
                    + 1 * np.einsum('ab,uv,tv->atbu', id_vir, self.dm1_cas, self.F_core[ncore:nocc, ncore:nocc],
                                    optimize="optimal"))
            Htmp[nocc:, ncore:nocc, nocc:, ncore:nocc] += 2 * np.einsum('tu,ab->atbu', self.dm1_cas,
                                                                        self.F_core[nocc:, nocc:], optimize="optimal")

        # active-core virtual-active H_{ti,au}
        if ncore > 0 and nvir > 0:
            avti = self.ppoo[nocc:, ncore:nocc, ncore:nocc, :ncore]
            aitv = self.ppoo[nocc:, :ncore, ncore:nocc, ncore:nocc]

            Htmp[ncore:nocc, :ncore, nocc:, ncore:nocc] = (- 4 * np.einsum('tuvx,aivx->tiau', 0.5 * self.dm2_cas,
                                                                           self.ppoo[nocc:, :ncore, ncore:nocc,
                                                                           ncore:nocc], optimize="optimal")
                                                           - 4 * np.einsum('tvux,axvi->tiau', 0.5 * self.dm2_cas,
                                                                           self.ppoo[nocc:, ncore:nocc, ncore:nocc,
                                                                           :ncore], optimize="optimal")
                                                           - 4 * np.einsum('tvxu,axvi->tiau', 0.5 * self.dm2_cas,
                                                                           self.ppoo[nocc:, ncore:nocc, ncore:nocc,
                                                                           :ncore], optimize="optimal"))
            Htmp[ncore:nocc, :ncore, nocc:, ncore:nocc] += (2 * np.einsum('uv,avti->tiau', self.dm1_cas,
                                                                          4 * avti - aitv.transpose(
                                                                              (0, 3, 2, 1)) - avti.transpose(
                                                                              (0, 2, 1, 3)), optimize="optimal")
                                                            - 2 * np.einsum('tu,ai->tiau', self.dm1_cas,
                                                                            self.F_core[nocc:, :ncore],
                                                                            optimize="optimal")
                                                            + 2 * np.einsum('tu,ai->tiau', id_cas, F_tot[nocc:, :ncore],
                                                                            optimize="optimal"))

            # virtual-active active-core  H_{au,ti}
            Htmp[nocc:, ncore:nocc, ncore:nocc, :ncore] = np.einsum('auti->tiau',
                                                                    Htmp[ncore:nocc, :ncore, nocc:, ncore:nocc],
                                                                    optimize="optimal")

        # active-core active-core H_{ti,uj} Nick 18 Mar
        if ncore > 0:
            gixyj = self.popo[:ncore, ncore:nocc, ncore:nocc, :ncore]
            gtijx = self.popo[ncore:nocc, :ncore, :ncore, ncore:nocc]
            gtxji = self.popo[ncore:nocc, ncore:nocc, :ncore, :ncore]
            gtwxy = self.popo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc]
            tiuj = 2 * np.einsum("tu,ij->tiuj", self.dm1_cas, self.F_core[:ncore, :ncore]) + \
                   4 * np.einsum("ij,tu->tiuj", id_cor, F_tot[ncore:nocc, ncore:nocc]) - \
                   np.einsum("ij,tw,uw->tiuj", id_cor, self.dm1_cas, self.F_core[ncore:nocc, ncore:nocc]) - \
                   np.einsum("ij,twxy,uwxy->tiuj", id_cor, self.dm2_cas, gtwxy) - \
                   np.einsum("ij,uw,tw->tiuj", id_cor, self.dm1_cas, self.F_core[ncore:nocc, ncore:nocc]) - \
                   np.einsum("ij,uwxy,twxy->tiuj", id_cor, self.dm2_cas, gtwxy) - \
                   2 * np.einsum("tu,ji->tiuj", id_cas, F_tot[:ncore, :ncore]) - \
                   2 * np.einsum("tu,ij->tiuj", id_cas, F_tot[:ncore, :ncore]) + \
                   2 * np.einsum("xu,tijx->tiuj", id_cas - self.dm1_cas,
                                4 * gtijx - gtijx.transpose((0,2,1,3)) - gtxji.transpose((0,3,2,1))) + \
                   2 * np.einsum("xt,xiju->tiuj", id_cas - self.dm1_cas,
                                4 * gtijx - gtijx.transpose((0,2,1,3)) - gtxji.transpose((0,3,2,1))) + \
                   2 * np.einsum("txuy,ixyj->tiuj", self.dm2_cas, gixyj) + \
                   2 * np.einsum("txyu,ixyj->tiuj", self.dm2_cas, gixyj) + \
                   2 * np.einsum("tuxy,ijxy->tiuj", self.dm2_cas, gtxji.transpose((3,2,1,0)))
            Htmp[ncore:nocc, :ncore, ncore:nocc, :ncore] = tiuj
            Htmp[ncore:nocc, :ncore, ncore:nocc, :ncore] = np.einsum("tiuj->ujti", tiuj)

#        # active-core active-core H_{ti,uj}
#        if ncore > 0:
#            viuj = self.ppoo[ncore:nocc, :ncore, ncore:nocc, :ncore]
#            uvij = self.ppoo[ncore:nocc, ncore:nocc, :ncore, :ncore]
#
#            Htmp[ncore:nocc, :ncore, ncore:nocc, :ncore] = 2 * np.einsum('tv,viuj->tiuj', id_cas - self.dm1_cas,
#                                                                         4 * viuj - viuj.transpose(
#                                                                             (2, 1, 0, 3)) - uvij.transpose(
#                                                                             (1, 2, 0, 3)), optimize="optimal")
#            Htmp[ncore:nocc, :ncore, ncore:nocc, :ncore] += np.einsum('tiuj->uitj',
#                                                                      Htmp[ncore:nocc, :ncore, ncore:nocc, :ncore],
#                                                                      optimize="optimal")
#            Htmp[ncore:nocc, :ncore, ncore:nocc, :ncore] += (4 * np.einsum('utvx,vxij->tiuj', 0.5 * self.dm2_cas,
#                                                                           self.ppoo[ncore:nocc, ncore:nocc, :ncore,
#                                                                           :ncore], optimize="optimal")
#                                                             + 4 * np.einsum('uxvt,vixj->tiuj', 0.5 * self.dm2_cas,
#                                                                             self.ppoo[ncore:nocc, :ncore, ncore:nocc,
#                                                                             :ncore], optimize="optimal")
#                                                             + 4 * np.einsum('uxtv,vixj->tiuj', 0.5 * self.dm2_cas,
#                                                                             self.ppoo[ncore:nocc, :ncore, ncore:nocc,
#                                                                             :ncore], optimize="optimal"))
#            Htmp[ncore:nocc, :ncore, ncore:nocc, :ncore] += (
#                    2 * np.einsum('tu,ij->tiuj', self.dm1_cas, self.F_core[:ncore, :ncore], optimize="optimal")
#                    - 4 * np.einsum('ij,tvxy,uvxy->tiuj', id_cor, 0.5 * self.dm2_cas,
#                                    self.ppoo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc], optimize="optimal")
#                    - 2 * np.einsum('ij,uv,tv->tiuj', id_cor, self.dm1_cas, self.F_core[ncore:nocc, ncore:nocc],
#                                    optimize="optimal"))
#            Htmp[ncore:nocc, :ncore, ncore:nocc, :ncore] += (
#                    4 * np.einsum('ij,tu->tiuj', id_cor, F_tot[ncore:nocc, ncore:nocc], optimize="optimal")
#                    - 4 * np.einsum('tu,ij->tiuj', id_cas, F_tot[:ncore, :ncore], optimize="optimal"))
#
#            # AM: I need to think about this
#            Htmp[ncore:nocc, :ncore, ncore:nocc, :ncore] = 0.5 * (
#                    Htmp[ncore:nocc, :ncore, ncore:nocc, :ncore] + np.einsum('tiuj->ujti',
#                                                                             Htmp[ncore:nocc, :ncore, ncore:nocc,
#                                                                             :ncore], optimize="optimal"))
        # Nick: Active-active Hessian contributions
        # active-active active-core H_{xy,ti}
        if ncore > 0:
            gxvit = self.ppoo[ncore:nocc, ncore:nocc, :ncore, ncore:nocc]
            gxivt = self.popo[ncore:nocc, :ncore, ncore:nocc, ncore:nocc]
            gxtiv = self.ppoo[ncore:nocc, ncore:nocc, :ncore, ncore:nocc]
            xyti = 2 * np.einsum("xt,yi->xyti", self.dm1_cas, self.F_core[ncore:nocc, :ncore], optimize="optimal") + \
                   2 * np.einsum("xvtw,yvwi->xyti", self.dm2_cas, self.popo[ncore:nocc, ncore:nocc, ncore:nocc, :ncore],
                             optimize="optimal") + \
                   2 * np.einsum("xvwt,yvwi->xyti", self.dm2_cas, self.popo[ncore:nocc, ncore:nocc, ncore:nocc, :ncore],
                             optimize="optimal") + \
                   2 * np.einsum("xtvw,yivw->xyti", self.dm2_cas, self.popo[ncore:nocc, :ncore, ncore:nocc, ncore:nocc],
                             optimize="optimal") + \
                   2 * np.einsum("yv,xvit->xyti", self.dm1_cas,
                             4 * gxvit - gxivt.transpose((0, 2, 1, 3)) - gxtiv.transpose((0, 3, 2, 1)),
                             optimize="optimal") + \
                   np.einsum("yt,xw,iw->xyti", id_cas, self.dm1_cas, self.F_core[:ncore, ncore:nocc],
                             optimize="optimal") + \
                   np.einsum("yt,xuwz,iuwz->xyti", id_cas, self.dm2_cas,
                             self.popo[:ncore, ncore:nocc, ncore:nocc, ncore:nocc],
                             optimize="optimal") + \
                   2 * np.einsum("xi,yt->xyti", F_tot[ncore:nocc, :ncore], id_cas, optimize="optimal")
            Htmp[ncore:nocc, ncore:nocc, ncore:nocc, :ncore] = xyti - np.einsum("xyti->yxti", xyti)
        # active-core active-active H_{ti, xy}
        if ncore > 0:
            Htmp[ncore:nocc, :ncore, ncore:nocc, ncore:nocc] = np.einsum("xyti->tixy",
                                                                         Htmp[ncore:nocc, ncore:nocc, ncore:nocc,
                                                                         :ncore])

        # active-active virtual-core H_{xy,ai}, as well as virtual-core active-active H_{ai,xy}
        if ncore > 0 and nvir > 0:
            gyvai = self.popo[ncore:nocc, ncore:nocc, nocc:, :ncore]
            gyiav = self.popo[ncore:nocc, :ncore, nocc:, ncore:nocc]
            gayiv = self.popo[nocc:, ncore:nocc, :ncore, ncore:nocc]
            Yxyia = 2 * np.einsum("xv,yvai->xyai", self.dm1_cas,
                                  4 * gyvai - gyiav.transpose((0, 3, 2, 1)) - gayiv.transpose((1, 3, 0, 2)),
                                  optimize="optimal")
            Htmp[ncore:nocc, ncore:nocc, nocc:, :ncore] = -Yxyia + np.einsum("xyai->yxai", Yxyia)
            Htmp[nocc:, :ncore, ncore:nocc, ncore:nocc] = np.einsum("xyai->aixy",Htmp[ncore:nocc, ncore:nocc, nocc:, :ncore])

        # active-active virtual-active H_{xy,at}
        if nvir > 0:
            xyat = 2 * np.einsum("yt,xa->xyat", self.dm1_cas, self.F_core[ncore:nocc, nocc:], optimize="optimal") + \
                   np.einsum("xt,aw,yw->xyat", id_cas, self.F_core[nocc:, ncore:nocc], self.dm1_cas, optimize="optimal") + \
                   np.einsum("xt,yuwz,auwz->xyat", id_cas, self.dm2_cas,self.popo[nocc:, ncore:nocc, ncore:nocc, ncore:nocc], optimize="optimal") + \
                   2 * np.einsum("yvtw,xvaw->xyat", self.dm2_cas, self.popo[ncore:nocc, ncore:nocc, nocc:, ncore:nocc], optimize="optimal") + \
                   2 * np.einsum("yvwt,xvaw->xyat", self.dm2_cas, self.popo[ncore:nocc, ncore:nocc, nocc:,ncore:nocc], optimize="optimal") + \
                   2 * np.einsum("ytvw,axvw->xyat", self.dm2_cas, self.popo[nocc:, ncore:nocc, ncore:nocc, ncore:nocc], optimize="optimal")
            Htmp[ncore:nocc, ncore:nocc, nocc:, ncore:nocc] = xyat - np.einsum("xyat->yxat", xyat)

        # virtual-active active-active H_{at, xy}
        if nvir > 0:
            Htmp[nocc:, ncore:nocc, ncore:nocc, ncore:nocc] = np.einsum("xyat->atxy",
                                                                        Htmp[ncore:nocc, ncore:nocc, nocc:, ncore:nocc])

        # active-active active-active H_{xy,tv}
        xytv = 2 * np.einsum("xt,yv->xytv", self.dm1_cas, self.F_core[ncore:nocc, ncore:nocc], optimize="optimal") + \
               2 * np.einsum("xwtz,ywzv->xytv", self.dm2_cas, self.popo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc],
                             optimize="optimal") + \
               2 * np.einsum("xwzt,ywzv->xytv", self.dm2_cas, self.popo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc],
                             optimize="optimal") + \
               2 * np.einsum("xtwz,yvwz->xytv", self.dm2_cas, self.popo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc],
                             optimize="optimal") - \
               np.einsum("yv,xw,tw->xytv", id_cas, self.dm1_cas, self.F_core[ncore:nocc, ncore:nocc],
                         optimize="optimal") - \
               np.einsum("yv,tw,xw->xytv", id_cas, self.dm1_cas, self.F_core[ncore:nocc, ncore:nocc],
                         optimize="optimal") - \
               np.einsum("yv,xuwz,tuwz->xytv", id_cas, self.dm2_cas,
                         self.popo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc],
                         optimize="optimal") - \
               np.einsum("yv,tuwz,xuwz->xytv", id_cas, self.dm2_cas,
                         self.popo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc],
                         optimize="optimal")
        Htmp[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc] = xytv - \
                                                               np.einsum("xytv->yxtv", xytv) - \
                                                               np.einsum("xytv->xyvt", xytv) + \
                                                               np.einsum("xytv->yxvt", xytv)
        return (Htmp)

    def _eig(self, h, *args):
        return scf.hf.eig(h, None)

    def get_hcore(self, mol=None):
        return self.hcore

    def get_fock(self, mo_coeff=None, ci=None, eris=None, casdm1=None, verbose=None):
        return mcscf.casci.get_fock(self, mo_coeff, ci, eris, casdm1, verbose)

    def cas_natorb(self, mo_coeff=None, ci=None, eris=None, sort=False, casdm1=None, verbose=None,
                   with_meta_lowdin=True):
        test = mcscf.casci.cas_natorb(self, mo_coeff, ci, eris, sort, casdm1, verbose, True)
        return test

    def edit_mask_by_gcoupling(self, mask):
        r"""
        This function looks at the genealogical coupling scheme and modifies a given mask.
        The mask restricts the number of free parameters.

        The algorithm works by looking at each column and traversing downwards the columns.
        """
        #n_dim = mask.shape[0]
        g_coupling_arr = list(self.g_coupling)
        n_dim = len(g_coupling_arr)
        for i, gfunc in enumerate(g_coupling_arr):    # This is for the columns
            for j in range(i+1, n_dim):    # This is for the rows
                if gfunc == g_coupling_arr[j]:
                    mask[j,i] = False
                else:
                    break
        return mask


    def uniq_var_indices(self, nmo, frozen):
        ''' This function creates a matrix of boolean of size (norb,norb).
            A True element means that this rotation should be taken into
            account during the optimization. Taken from pySCF.mcscf.casscf '''
        nocc = self.ncore + self.ncas
        mask = np.zeros((self.norb, self.norb), dtype=bool)
        mask[self.ncore:nocc, :self.ncore] = True  # Active-Core rotations
        mask[nocc:, :nocc] = True                  # Virtual-Core and Virtual-Active rotations
        mask[self.ncore:nocc, self.ncore:nocc] = np.tril(np.ones((self.ncas,self.ncas),dtype=bool),k=-1)  # Active-Active rotations
        if self.g_coupling is not None:
            mask[self.ncore:nocc, self.ncore:nocc] = self.edit_mask_by_gcoupling(mask[self.ncore:nocc, self.ncore:nocc])    # Make use of genealogical coupling to remove degrees of freedom
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

    @property
    def orbital_energies(self):
        r"""
        Gets the orbital energies (This is found by diagonalising the Fock matrix)
        """
        evals, evecs = np.linalg.eigh(self.F_core + self.F_cas)
        return evals

    def canonicalise(self):
        r"""
        Forms the canonicalised MO coefficients by diagonalising invariant subblocks of the Fock matrix
        """
        #print("Canonicalising orbitals")
        # Build Fock matrix
        F_tot = self.F_core + self.F_cas
        # Get Core-Core, Active-Active (if exists from g-coupling pattern), and Virtual-Virtual
        nocc = self.ncore + self.ncas
        F_cc = F_tot[:self.ncore, :self.ncore]
        F_vv = F_tot[nocc:, nocc:]
        F_aa = np.identity(F_tot[self.ncore:nocc, self.ncore:nocc].shape)
        # Add Fock matrix subblocks into a list
        Fs = [F_cc]
        #active_partition = self.get_gcoupling_partition()
        #for i in range(len(active_partition)-1):
        #    Ftemp = F_aa[active_partition[i]:active_partition[i+1], active_partition[i]:active_partition[i+1]]
        #    Fs.append(Ftemp)
        Fs.append(F_vv)
        # Diagonalise each matrix and build transformation matrix
        Us = []
        for i, f in enumerate(Fs):
            evals, evecs = np.linalg.eigh(f)
            Us.append(evecs)
        U = scipy.linalg.block_diag(*Us)
        # Transform coefficients
        self.mo_coeff = self.mo_coeff @ U

    def CASRDM1_to_RDM1(self, dm1_cas, transition=False):
        ''' Transform 1-RDM from CAS space into full MO space'''
        ncore = self.ncore
        ncas = self.ncas
        dm1 = np.zeros((self.norb, self.norb))
        if transition is False and self.ncore > 0:
            dm1[:ncore, :ncore] = 2 * np.identity(self.ncore, dtype="int")
        dm1[ncore:ncore + ncas, ncore:ncore + ncas] = dm1_cas
        return dm1

    def CASRDM2_to_RDM2(self, dm1_cas, dm2_cas, transition=False):
        ''' This method takes a 2-RDM in the CAS space and transform it to the full MO space '''
        ncore = self.ncore
        ncas = self.ncas
        norb = self.norb
        nocc = ncore + ncas

        dm1 = self.CASRDM1_to_RDM1(dm1_cas, transition)
        dm2 = np.zeros((norb, norb, norb, norb))
        if transition is False:
            # Core contributions
            for i in range(ncore):
                for j in range(ncore):
                    for k in range(ncore):
                        for l in range(ncore):
                            dm2[i, j, k, l] = 4 * delta_kron(i, j) * delta_kron(k, l)
                            dm2[i, j, k, l] -= 2 * delta_kron(i, l) * delta_kron(k, j)

                    for p in range(ncore, nocc):
                        for q in range(ncore, nocc):
                            dm2[i, j, p, q] = 2 * delta_kron(i, j) * dm1[q, p]
                            dm2[p, q, i, j] = dm2[i, j, p, q]

                            dm2[i, q, p, j] = - delta_kron(i, j) * dm1[q, p]
                            dm2[p, j, i, q] = dm2[i, q, p, j]

        else:
            for i in range(ncore):
                for j in range(ncore):
                    for p in range(ncore, ncore + ncas):
                        for q in range(ncore, ncore + ncas):
                            dm2[i, j, p, q] = 2 * delta_kron(i, j) * dm1[q, p]
                            dm2[p, q, i, j] = dm2[i, j, p, q]

                            dm2[i, q, p, j] = - delta_kron(i, j) * dm1[q, p]
                            dm2[p, j, i, q] = dm2[i, q, p, j]

        # Insert the active-active sector
        dm2[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc] = dm2_cas
        return dm2

    def get_numerical_gradient(self, eps=1e-3):
        grad = np.zeros((self.dim))
        for i in range(self.dim):
            x1 = np.zeros(self.dim)
            x2 = np.zeros(self.dim)

            x1[i] += eps
            x2[i] -= eps

            self.take_step(x1)
            E1 = self.energy
            self.restore_last_step()

            self.take_step(x2)
            E2 = self.energy
            self.restore_last_step()

            grad[i] = (E1 - E2) / (2 * eps)

        return grad

    def get_numerical_hessian(self, eps=1e-3):
        Hess = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            print(i, self.dim)
            for j in range(i, self.dim):
                x1 = np.zeros(self.dim)
                x2 = np.zeros(self.dim)
                x3 = np.zeros(self.dim)
                x4 = np.zeros(self.dim)

                x1[i] += eps
                x1[j] += eps
                x2[i] += eps
                x2[j] -= eps
                x3[i] -= eps
                x3[j] += eps
                x4[i] -= eps
                x4[j] -= eps

                self.take_step(x1)
                E1 = self.energy
                self.restore_last_step()

                self.take_step(x2)
                E2 = self.energy
                self.restore_last_step()

                self.take_step(x3)
                E3 = self.energy
                self.restore_last_step()

                self.take_step(x4)
                E4 = self.energy
                self.restore_last_step()

                Hess[i, j] = ((E1 - E2) - (E3 - E4)) / (4 * eps * eps)
                if (i != j): Hess[j, i] = Hess[i, j]

        return Hess