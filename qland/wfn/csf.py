#!/usr/bin/python3
# Modified from ss_casscf code of Antoine Marie and Hugh G. A. Burton
# This is code for a CSF, which can be formed in a variety of ways.

import numpy as np
import scipy
from functools import reduce
from typing import List
from pygnme import wick, utils, owndata
from pyscf import scf, fci, __config__, ao2mo, lib, mcscf
from qland.io.parser import getlist, getvalue
from qland.wfn.CSFBuilder.CGCSF import CGCSF
from qland.wfn.CSFBuilder.GCCSF import GCCSF
from qland.wfn.CSFBuilder.NoCSF import NoCSF
from qland.wfn.CSFBuilder.Operators.Operators import get_generic_no_overlap
from qland.utils.linalg import delta_kron, orthogonalise
from .wavefunction import Wavefunction


class CSF(Wavefunction):
    def __init__(self, mol, stot, active_space=None, core: List[int] = None,
                 active: List[int] = None, g_coupling: str = None, frozen: int = 0,
                 permutation: List[int] = None, mo_basis: 'str' = 'site', csf_build: str = 'genealogical',
                 localstots: List[float] = None, active_subspaces: List[int] = None):

        # Save molecule and reference SCF
        self.mol = mol
        self._scf = scf.RHF(mol)
        self.max_memory = self._scf.max_memory

        # Get AO integrals
        self.get_ao_integrals()

        self.frozen = frozen

        # Setup CSF variables
        if csf_build.lower() == 'nocsf':
            self.setup_nocsf(stot, core, mo_basis, localstots, active_subspaces)
        else:
            self.setup_csf(stot, active_space, core, active, g_coupling, permutation, mo_basis, csf_build,
                           localstots, active_subspaces)

        # Get number of determinants (which is the dimension of the problem)
        # self.nDet = self.csf_instance.n_dets

        # Save mapping indices for unique orbital rotations
        self.frozen = frozen


    @property
    def dim(self):
        """Number of degrees of freedom"""
        return self.nrot

    @property
    def energy(self):
        ''' Compute the energy corresponding to a given set of
             one-el integrals, two-el integrals, 1- and 2-RDM '''
        E = self.energy_core
        if self.dm1_cas is not None:
            E += np.einsum('pq,pq', self.h1eff, self.dm1_cas, optimize="optimal")
        if self.dm2_cas is not None:
            E += 0.5 * np.einsum('pqrs,pqrs', self.h2eff, self.dm2_cas, optimize="optimal")
        return E

    @property
    def s2(self):
        ''' Compute the spin of a given FCI vector '''
        return self.csf_instance.get_s2()

    @property
    def gradient(self):
        r"""
        Unlike in CAS case, we are only interested in orbital optimisation
        :return:
        """
        return self.get_orbital_gradient()

    @property
    def hessian(self):
        ''' This method finds orb-orb part of the Hessian '''
        return (self.get_hessianOrbOrb()[:, :, self.rot_idx])[self.rot_idx, :]

    def setup_csf(self, stot: float, active_space: List[int], core: List[int], active: List[int],
                  g_coupling: str, permutation: List[int], mo_basis: 'str' = "site",
                  csf_build: str = 'genealogical', localstots: List[float] = None,
                  active_subspaces: List[int] = None):

        self.ncas = active_space[0]  # Number of active orbitals
        self.stot = stot  # Total S value
        self.core = core  # List of core (doubly occupied) orbitals
        self.act = active  # List of active orbitals

        self.g_coupling = g_coupling  # Genealogical coupling pattern
        self.permutation = permutation  # Permutation for spin coupling

        self.mo_basis = mo_basis  # Choice of basis for orbital guess
        self.csf_build = csf_build  # Method for constructing CSFs
        self.localstots = localstots  # Local spins
        self.active_subspaces = active_subspaces  # Local active spaces
        self.mat_ci = None

        if isinstance(active_space[1], (int, np.integer)):
            nelecb = (active_space[1] - self.mol.spin) // 2
            neleca = active_space[1] - nelecb
            self.nelecas = (neleca, nelecb)  # Tuple of number of active electrons
        else:
            self.nelecas = np.asarray((active_space[1][0], active_space[1][1])).astype(int)

        ncorelec = self.mol.nelectron - sum(self.nelecas)
        assert ncorelec % 2 == 0
        assert ncorelec >= 0
        self.ncore = ncorelec // 2

        # Build CSF
        if csf_build.lower() == 'genealogical':
            self.csf_instance = GCCSF(self.mol, self.stot, len(self.core), len(self.act),
                                      self.g_coupling, self.mo_basis)
        elif csf_build.lower() == 'clebschgordon':
            assert localstots is not None, "Local spin quantum numbers (localstots) undefined"
            assert active_subspaces is not None, "Active subspaces (active_subspaces) undefined"
            assert active_subspaces[0] + active_subspaces[2] == active_space[0], "Mismatched number of active orbitals"
            self.csf_instance = CGCSF(self.mol, self.stot, localstots[0], localstots[1],
                                      (active_subspaces[0], active_subspaces[1]),
                                      (active_subspaces[2], active_subspaces[3]),
                                      len(self.core), len(self.act))
        else:
            import sys
            sys.exit("The requested CSF build is not supported, exiting.")
        # Save mapping indices for unique orbital rotations
        self.rot_idx = self.uniq_var_indices(self.norb, self.frozen)
        self.nrot = np.sum(self.rot_idx)

    def setup_nocsf(self, stot: float, core: List[int], mo_basis: 'str' = "site",
                    localstots: List[float] = None, active_subspaces: List[int] = None):

        self.stot = stot  # Total S value
        self.core = core  # List of core (doubly occupied) orbitals
        self.act = []
        self.mo_basis = mo_basis  # Choice of basis for orbital guess
        self.csf_build = 'nocsf'  # Method for constructing CSFs
        self.localstots = localstots  # Local spins
        self.active_subspaces = active_subspaces  # Local active spaces
        self.mat_ci = None

        ncorelec = self.mol.nelectron
        assert ncorelec % 2 == 0
        assert ncorelec >= 0
        self.nelecas = (0,0)
        self.ncore = ncorelec // 2
        self.ncas = 0
        self.g_coupling = None
        self.permutation = None

        # Build NoCSF
        self.csf_instance = NoCSF(self.mol, self.stot, len(self.core), self.mo_basis)
        # Save mapping indices for unique orbital rotations
        self.rot_idx = self.uniq_var_indices(self.norb, self.frozen)
        self.nrot = np.sum(self.rot_idx)

    def save_to_disk(self, tag):
        """Save a CSF to disk with prefix 'tag'"""
        # Get the Hessian index
        hindices = self.get_hessian_index()

        # self.mo_coeff = permute_out_orbitals(self.mo_coeff)
        # Save coefficients and energy
        np.savetxt(tag + '.mo_coeff', self.mo_coeff, fmt="% 20.16f")
        np.savetxt(tag + '.energy',
                   np.array([[self.energy, hindices[0], hindices[1], self.s2]]),
                   fmt="% 18.12f % 5d % 5d % 12.6f")

        # Save CSF config
        with open(tag + '.csf', 'w') as outF:
            outF.write('total_spin             {:2.2f}\n'.format(self.stot))
            outF.write('active_space           {:d} {:d}\n'.format(self.ncas, self.nelecas[0] + self.nelecas[1]))
            outF.write(('core_orbitals         ' + len(self.core) * " {:d}" + "\n").format(*self.core))
            outF.write(('active_orbitals       ' + len(self.act) * " {:d}" + "\n").format(*self.act))
            outF.write('genealogical_coupling  {:s}\n'.format(self.g_coupling))
            outF.write(('coupling_permutation  ' + len(self.permutation) * " {:d}" + "\n").format(*self.permutation))
            outF.write('csf_build              {:s}\n'.format(self.csf_build))
            outF.write(('localstots            ' + len(self.localstots) * " {:d}" + "\n").format(*self.localstots))
            outF.write(
                ('active_subspaces      ' + len(self.active_subspaces) * " {:d}" + "\n").format(*self.active_subspaces))

    def read_from_disk(self, tag):
        """Read a SS-CASSCF object from disk with prefix 'tag'"""
        # Read MO coefficient and CI coefficients
        mo_coeff = np.genfromtxt(tag + ".mo_coeff")

        # Read the .csf file
        with open(tag + '.csf', 'r') as inF:
            lines = inF.read().splitlines()

        stot = getvalue(lines, "total_spin", float, True)
        active_space = getlist(lines, "active_space", int, False)
        core = getlist(lines, "core_orbitals", int, False)
        active = getlist(lines, "active_orbitals", int, False)
        permutation = getlist(lines, "coupling_permutation", int, False)
        g_coupling = getvalue(lines, "genealogical_coupling", str, False)
        csf_build = getvalue(lines, "csf_build", str, True)
        localstots = getlist(lines, "local_spins", float, False)
        active_subspaces = getlist(lines, "active_subspaces", int, False)

        # Setup CSF
        if csf_build.lower() == 'nocsf':
            self.setup_nocsf(stot, core, None, localstots, active_subspaces)
        else:
            self.setup_csf(stot, active_space, core, active, g_coupling, permutation, None, csf_build,
                           localstots, active_subspaces)
        # Initialise object
        self.initialise(mo_coeff, None)

    def copy(self):
        """Return a copy of the current object"""
        # When copying, keep the MO coefficients as they are
        newcsf = CSF(self.mol, self.stot, [self.ncas, self.nelecas], list(np.arange(len(self.core), dtype=int)),
                     list(np.arange(len(self.core), len(self.core) + len(self.act), dtype=int)),
                     self.g_coupling, self.frozen, self.permutation, self.mo_basis,
                     self.csf_build, self.localstots, self.active_subspaces)
        newcsf.initialise(self.mo_coeff, integrals=False)
        return newcsf

    def overlap(self, ref):
        r"""
        Determines the overlap between the current CSF and a reference CSF.

        :param ref: A ConfigurationStateFunction object which we are comparing to
        """
        #csf_coeffs = self.csf_instance.csf_coeffs
        #ref_coeffs = ref.csf_instance.csf_coeffs
        #smo = np.einsum("ip,ij,jq->pq", self.mo_coeff, self.ovlp, ref.mo_coeff)
        #cross_overlap_mat = scipy.linalg.block_diag(smo, smo)
        #return get_generic_no_overlap(self.csf_instance.dets_sq, ref.csf_instance.dets_sq, csf_coeffs, ref_coeffs,
        #                              cross_overlap_mat)
        s, h = self.hamiltonian(ref)
        return s

    def hamiltonian(self, other, thresh=1e-10):
        # x = self
        # w = other
        self.update_integrals()
        na = self.nelecas[0] + self.ncore
        nb = self.nelecas[1] + self.ncore
        h1e = owndata(self._scf.get_hcore())
        h2e = owndata(ao2mo.restore(1, self._scf._eri, self.mol.nao).reshape(self.mol.nao ** 2, self.mol.nao ** 2))

        # Setup biorthogonalised orbital pair
        refxa = wick.reference_state[float](self.nmo, self.nmo, na, self.ncas, self.ncore, owndata(self.mo_coeff))
        refxb = wick.reference_state[float](self.nmo, self.nmo, nb, self.ncas, self.ncore, owndata(self.mo_coeff))
        refwa = wick.reference_state[float](self.nmo, self.nmo, na, other.ncas, other.ncore, owndata(other.mo_coeff))
        refwb = wick.reference_state[float](self.nmo, self.nmo, nb, other.ncas, other.ncore, owndata(other.mo_coeff))

        # Setup paired orbitals
        orba = wick.wick_orbitals[float, float](refxa, refwa, owndata(self.ovlp))
        orbb = wick.wick_orbitals[float, float](refxb, refwb, owndata(self.ovlp))

        # Setup matrix builder object
        mb = wick.wick_uscf[float, float, float](orba, orbb, self.mol.energy_nuc())
        # Add one- and two-body contributions
        mb.add_one_body(h1e)
        mb.add_two_body(h2e)

        # Generate lists of FCI bitsets
        vxa = utils.fci_bitset_list(na - self.ncore, self.ncas)
        vxb = utils.fci_bitset_list(nb - self.ncore, self.ncas)
        vwa = utils.fci_bitset_list(na - other.ncore, other.ncas)
        vwb = utils.fci_bitset_list(nb - other.ncore, other.ncas)

        s = 0
        h = 0

        # Loop over FCI occupation strings
        for iwa in range(len(vwa)):
            for iwb in range(len(vwb)):
                if (abs(owndata(other.csf_instance.ci)[iwa, iwb]) < thresh):
                    # Skip if coefficient is below threshold
                    continue
                for ixa in range(len(vxa)):
                    for ixb in range(len(vxb)):
                        if (abs(owndata(self.csf_instance.ci)[ixa, ixb]) < thresh):
                            # Skip if coefficient is below threshold
                            continue
                        # Compute S and H contribution for this pair of determinants
                        stmp, htmp = mb.evaluate(vxa[ixa], vxb[ixb], vwa[iwa], vwb[iwb])
                        # Accumulate the Hamiltonian and overlap matrix elements
                        s += stmp * owndata(other.csf_instance.ci)[iwa, iwb] * owndata(self.csf_instance.ci)[ixa, ixb]
                        h += htmp * owndata(other.csf_instance.ci)[iwa, iwb] * owndata(self.csf_instance.ci)[ixa, ixb]
        return s, h

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

    def permute_in_orbitals(self, mo_coeff):
        """A method to permute the MO coefficients read in"""
        norbs = mo_coeff.shape[1]
        virs = list(set([i for i in range(norbs)]) - set(self.core + self.act))
        core_orbs = mo_coeff[:, self.core]
        if bool(self.permutation):
            act_orbs = mo_coeff[:, self.act][:, self.permutation]
        else:
            act_orbs = mo_coeff[:, self.act]
        vir_orbs = mo_coeff[:, virs]
        return np.hstack([core_orbs, act_orbs, vir_orbs])

    def permute_out_orbitals(self, mo_coeff):
        """A method to permute the MO coefficients returned"""
        naos = mo_coeff.shape[0]
        norbs = mo_coeff.shape[1]
        virs = list(set([i for i in range(norbs)]) - set(self.core + self.act))
        new_coeffs = np.zeros((naos, norbs))
        new_coeffs[:, self.core] = self.mo_coeff[:, np.arange(len(self.core))]
        if bool(self.permutation):
            inv_perm = np.zeros(len(self.permutation), dtype=int)
            for i, p in enumerate(self.permutation):
                inv_perm[p] = i
            new_coeffs[:, self.act] = self.mo_coeff[:, np.arange(len(self.core), len(self.core) + len(self.act))][:, inv_perm]
        else:
            new_coeffs[:, self.act] = self.mo_coeff[:, np.arange(len(self.core), len(self.core) + len(self.act))]
        new_coeffs[:, virs] = self.mo_coeff[:, np.arange(len(self.core) + len(self.act), norbs)]
        return new_coeffs

    def initialise(self, mo_guess=None, mat_ci=None, integrals=True):
        # Save orbital coefficients
        if (not (mo_guess is not None)): mo_guess = self.csf_instance.coeffs.copy()
        mo_guess = orthogonalise(mo_guess, self.ovlp)
        self.mo_coeff = self.permute_in_orbitals(mo_guess)
        self.nmo = self.mo_coeff.shape[1]

        # Initialise integrals
        if (integrals): self.update_integrals()

    def deallocate(self):
        # Reduce the memory footprint for storing
        self._scf._eri = None
        self.ppoo = None
        self.popo = None
        self.h1e = None
        self.h1eff = None
        self.h2eff = None
        self.F_core = None
        self.F_cas = None

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

        if self.csf_instance.n_act == 0:
            self.energy_core = self.get_energy_core()
            self.dm1_cas, self.dm2_cas = None, None
        else:
            assert self.csf_instance.n_act > 0
            # Effective Hamiltonians in CAS space
            self.h1eff, self.energy_core = self.get_h1eff()
            self.h2eff = self.get_h2eff()
            self.h2eff = ao2mo.restore(1, self.h2eff, self.ncas)
            # Reduced density matrices
            self.dm1_cas, self.dm2_cas = self.get_csfrdm_12()
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

    def get_energy_core(self):
        '''Get core energy in the case of no active orbitals'''
        ncas = self.ncas
        assert ncas == 0
        ncore = nocc = self.ncore

        # Get core and active orbital coefficients
        mo_core = self.mo_coeff[:, :ncore]

        # Core density matrix (in AO basis)
        self.core_dm = np.dot(mo_core, mo_core.T) * 2

        # Core energy
        energy_core = self.enuc
        energy_core += np.einsum('ij,ji', self.core_dm, self.hcore, optimize="optimal")
        energy_core += self.vhf_c[:ncore, :ncore].trace()
        return energy_core

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

    def get_csfrdm_12(self):
        r"""
        Get 1RDM and 2RDM from PySCF
        """
        # dm1_csf, dm2_csf = self.csf_instance.get_pyscf_rdms()
        return self.csf_instance.rdm1, self.csf_instance.rdm2

    def get_fock_matrices(self):
        ''' Compute the core part of the generalized Fock matrix '''
        ncore = self.ncore

        if self.csf_instance.n_act == 0:
            self.F_core = self.h1e + self.vhf_c
            return
        else:
            # Full Fock
            vj = np.empty((self.nmo, self.nmo))
            vk = np.empty((self.nmo, self.nmo))
            for i in range(self.nmo):
                vj[i] = np.einsum('ij,qij->q', self.dm1_cas, self.ppoo[i, :, ncore:, ncore:], optimize="optimal")
                vk[i] = np.einsum('ij,iqj->q', self.dm1_cas, self.popo[i, ncore:, :, ncore:], optimize="optimal")
            fock = self.h1e_mo + self.vhf_c + vj - vk * 0.5

            # Core contribution
            self.F_core = self.h1e + self.vhf_c

            # Active space contribution
            self.F_cas = fock - self.F_core

            return

    def get_generalised_fock(self, dm1_cas, dm2_cas):
        ''' This method finds the generalised Fock matrix with a different method '''
        ncore = self.ncore
        nocc = self.ncore + self.ncas
        F = np.zeros((self.nmo, self.nmo))
        if self.csf_instance.n_act == 0:
            F[:ncore, :] = 2 * self.F_core[:, :ncore].T
        else:
            F[:ncore, :] = 2 * (self.F_core[:, :ncore] + self.F_cas[:, :ncore]).T
            F[ncore:nocc, :] = np.einsum("nw,vw->vn", self.F_core[:, ncore:nocc], dm1_cas) + \
                               np.einsum("vwxy,nwxy->vn", dm2_cas, self.popo[:, ncore:nocc, ncore:nocc, ncore:nocc])
        return 2 * F.T

    def get_orbital_gradient(self):
        ''' This method builds the orbital part of the gradient '''
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
        if self.csf_instance.n_act == 0:
            F_tot = self.F_core
        else:
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
        if ncore > 0 and nvir > 0 and ncas > 0:
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
        if ncore > 0 and nvir > 0 and ncas > 0:
            Htmp[nocc:, ncore:nocc, nocc:, :ncore] = np.einsum('aibt->btai', Htmp[nocc:, :ncore, nocc:, ncore:nocc],
                                                               optimize="optimal")

        # virtual-core active-core H_{ai,tj}
        if ncore > 0 and nvir > 0 and ncas > 0:
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
        if ncore > 0 and nvir > 0 and ncas > 0:
            Htmp[ncore:nocc, :ncore, nocc:, :ncore] = np.einsum('aitj->tjai', Htmp[nocc:, :ncore, ncore:nocc, :ncore],
                                                                optimize="optimal")

        # virtual-active virtual-active H_{at,bu}
        if nvir > 0 and ncas > 0:
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
        if ncore > 0 and nvir > 0 and ncas > 0:
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
        if ncore > 0 and ncas > 0:
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
                                 4 * gtijx - gtijx.transpose((0, 2, 1, 3)) - gtxji.transpose((0, 3, 2, 1))) + \
                   2 * np.einsum("xt,xiju->tiuj", id_cas - self.dm1_cas,
                                 4 * gtijx - gtijx.transpose((0, 2, 1, 3)) - gtxji.transpose((0, 3, 2, 1))) + \
                   2 * np.einsum("txuy,ixyj->tiuj", self.dm2_cas, gixyj) + \
                   2 * np.einsum("txyu,ixyj->tiuj", self.dm2_cas, gixyj) + \
                   2 * np.einsum("tuxy,ijxy->tiuj", self.dm2_cas, gtxji.transpose((3, 2, 1, 0)))
            Htmp[ncore:nocc, :ncore, ncore:nocc, :ncore] = tiuj
            Htmp[ncore:nocc, :ncore, ncore:nocc, :ncore] = np.einsum("tiuj->ujti", tiuj)

        # Nick: Active-active Hessian contributions
        # active-active active-core H_{xy,ti}
        if ncore > 0 and ncas > 0:
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
        if ncore > 0 and ncas > 0:
            Htmp[ncore:nocc, :ncore, ncore:nocc, ncore:nocc] = np.einsum("xyti->tixy",
                                                                         Htmp[ncore:nocc, ncore:nocc, ncore:nocc,
                                                                         :ncore])

        # active-active virtual-core H_{xy,ai}, as well as virtual-core active-active H_{ai,xy}
        if ncore > 0 and nvir > 0 and ncas > 0:
            gyvai = self.popo[ncore:nocc, ncore:nocc, nocc:, :ncore]
            gyiav = self.popo[ncore:nocc, :ncore, nocc:, ncore:nocc]
            gayiv = self.popo[nocc:, ncore:nocc, :ncore, ncore:nocc]
            Yxyia = 2 * np.einsum("xv,yvai->xyai", self.dm1_cas,
                                  4 * gyvai - gyiav.transpose((0, 3, 2, 1)) - gayiv.transpose((1, 3, 0, 2)),
                                  optimize="optimal")
            Htmp[ncore:nocc, ncore:nocc, nocc:, :ncore] = -Yxyia + np.einsum("xyai->yxai", Yxyia)
            Htmp[nocc:, :ncore, ncore:nocc, ncore:nocc] = np.einsum("xyai->aixy",
                                                                    Htmp[ncore:nocc, ncore:nocc, nocc:, :ncore])

        # active-active virtual-active H_{xy,at}
        if nvir > 0 and ncas > 0:
            xyat = 2 * np.einsum("yt,xa->xyat", self.dm1_cas, self.F_core[ncore:nocc, nocc:], optimize="optimal") + \
                   np.einsum("xt,aw,yw->xyat", id_cas, self.F_core[nocc:, ncore:nocc], self.dm1_cas,
                             optimize="optimal") + \
                   np.einsum("xt,yuwz,auwz->xyat", id_cas, self.dm2_cas,
                             self.popo[nocc:, ncore:nocc, ncore:nocc, ncore:nocc], optimize="optimal") + \
                   2 * np.einsum("yvtw,xvaw->xyat", self.dm2_cas, self.popo[ncore:nocc, ncore:nocc, nocc:, ncore:nocc],
                                 optimize="optimal") + \
                   2 * np.einsum("yvwt,xvaw->xyat", self.dm2_cas, self.popo[ncore:nocc, ncore:nocc, nocc:, ncore:nocc],
                                 optimize="optimal") + \
                   2 * np.einsum("ytvw,axvw->xyat", self.dm2_cas, self.popo[nocc:, ncore:nocc, ncore:nocc, ncore:nocc],
                                 optimize="optimal")
            Htmp[ncore:nocc, ncore:nocc, nocc:, ncore:nocc] = xyat - np.einsum("xyat->yxat", xyat)

        # virtual-active active-active H_{at, xy}
        if nvir > 0 and ncas > 0:
            Htmp[nocc:, ncore:nocc, ncore:nocc, ncore:nocc] = np.einsum("xyat->atxy",
                                                                        Htmp[ncore:nocc, ncore:nocc, nocc:, ncore:nocc])

        # active-active active-active H_{xy,tv}
        if ncas > 0:
            xytv = 2 * np.einsum("xt,yv->xytv", self.dm1_cas, self.F_core[ncore:nocc, ncore:nocc], optimize="optimal") + \
                   2 * np.einsum("xwtz,ywzv->xytv", self.dm2_cas,
                                 self.popo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc],
                                 optimize="optimal") + \
                   2 * np.einsum("xwzt,ywzv->xytv", self.dm2_cas,
                                 self.popo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc],
                                 optimize="optimal") + \
                   2 * np.einsum("xtwz,yvwz->xytv", self.dm2_cas,
                                 self.popo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc],
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
        # n_dim = mask.shape[0]
        g_coupling_arr = list(self.g_coupling)
        n_dim = len(g_coupling_arr)
        for i, gfunc in enumerate(g_coupling_arr):  # This is for the columns
            for j in range(i + 1, n_dim):  # This is for the rows
                if gfunc == g_coupling_arr[j]:
                    mask[j, i] = False
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
        mask[nocc:, :nocc] = True  # Virtual-Core and Virtual-Active rotations
        mask[self.ncore:nocc, self.ncore:nocc] = np.tril(np.ones((self.ncas, self.ncas), dtype=bool),
                                                         k=-1)  # Active-Active rotations
        if self.g_coupling is not None:
            mask[self.ncore:nocc, self.ncore:nocc] = self.edit_mask_by_gcoupling(mask[self.ncore:nocc,
                                                                                 self.ncore:nocc])  # Make use of genealogical coupling to remove degrees of freedom
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
        # print("Canonicalising orbitals")
        # Build Fock matrix
        F_tot = self.F_core + self.F_cas
        # Get Core-Core, Active-Active (if exists from g-coupling pattern), and Virtual-Virtual
        nocc = self.ncore + self.ncas
        F_cc = F_tot[:self.ncore, :self.ncore]
        F_vv = F_tot[nocc:, nocc:]
        F_aa = np.identity(F_tot[self.ncore:nocc, self.ncore:nocc].shape)
        # Add Fock matrix subblocks into a list
        Fs = [F_cc]
        # active_partition = self.get_gcoupling_partition()
        # for i in range(len(active_partition)-1):
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
