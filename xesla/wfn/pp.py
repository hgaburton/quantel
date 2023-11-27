#!/usr/bin/python3
# Author: Antoine Marie, Hugh G. A. Burton

import numpy as np
import scipy.linalg
from functools import reduce
from pyscf import scf, __config__, ao2mo, lo
from xesla.utils.linalg import delta_kron, orthogonalise
from .wavefunction import Wavefunction
from xesla.gnme.pcid_noci import pcid_coupling

class PP(Wavefunction):
    def __init__(self, mol):
        self.mol        = mol
        self.nelec      = mol.nelec
        self._scf       = scf.RHF(mol)
        self.verbose    = mol.verbose
        self.stdout     = mol.stdout
        self.max_memory = self._scf.max_memory
        # Get number of electrons
        self.na         = self.nelec[0]
        self.nb         = self.nelec[1]
        assert(self.na == self.nb)
        # Get AO integrals 
        self.get_ao_integrals()
        self.norb       = self.hcore.shape[0]
        self.nocc       = self.na
        self.nvir       = self.norb - self.na

        # Initialise space for amplitudes
        self.t = np.zeros((self.nocc,self.nvir))
        self.z = np.zeros((self.nocc,self.nvir))

        self.amp_idx = np.zeros((self.nocc,self.nvir),dtype=bool)
        for i in range(self.nocc):
            self.amp_idx[self.nocc-i-1,i] = True

        self.rot_idx    = self.uniq_var_indices()
        self.nrot       = np.sum(self.rot_idx)

    def get_ao_integrals(self):
        """Compute the required AO integrals"""
        self.enuc       = self._scf.energy_nuc()
        self.v1e        = self.mol.intor('int1e_nuc')  # Nuclear repulsion matrix elements
        self.t1e        = self.mol.intor('int1e_kin')  # Kinetic energy matrix elements
        self.hcore      = self.t1e + self.v1e          # 1-electron matrix elements in the AO basis
        self.norb       = self.hcore.shape[0]
        self.ovlp       = self.mol.intor('int1e_ovlp') # Overlap matrix
        self._scf._eri  = self.mol.intor("int2e", aosym="s8") # Two electron integrals

    def uniq_var_indices(self):
        ''' This function creates a matrix of boolean of size (norb,norb). 
            A True element means that this rotation should be taken into 
            account during the optimization. Taken from pySCF.mcscf.casscf '''
        mask = np.tril(np.ones(self.norb,dtype=bool),k=-1)
        return mask

    def update_integrals(self):
        # One-electron Hamiltonian
        self.h1e = np.einsum('ip,ij,jq->pq', self.mo_coeff, self.hcore, self.mo_coeff, optimize="optimal")

        # Occupied orbitals
        self.pppp = ao2mo.incore.general(self._scf._eri, (self.mo_coeff, self.mo_coeff, self.mo_coeff, self.mo_coeff), compact=False)
        self.pppp = self.pppp.reshape((self.nmo,self.nmo,self.nmo,self.nmo)).transpose(0,1,2,3)

        # Get reference energy
        self.E0  = self.enuc
        self.E0 += 2 * np.einsum('ii', self.h1e[:self.nocc, :self.nocc])
        self.E0 += 2 * np.einsum('iijj', self.pppp[:self.nocc, :self.nocc, :self.nocc, :self.nocc])
        self.E0 -= 1 * np.einsum('ijji', self.pppp[:self.nocc, :self.nocc, :self.nocc, :self.nocc])

        # Get Fock matrix
        self.F  = np.copy(self.h1e)
        self.F += 2 * np.einsum('pqjj->pq', self.pppp[:,:,:self.nocc,:self.nocc]) - np.einsum('pjjq->pq', self.pppp[:,:self.nocc,:self.nocc,:])


    def get_pair_indices(self, ie):
        """Get indices for occupied-virtual orbital pair"""
        return self.nocc - ie - 1, ie + self.nocc

    def solve_amplitudes(self):
        """Analytically solve the perfect pairing amplitudes"""
        told = self.t.copy()

        self.Ec = 0
        for ie in range(self.nocc):
            io, iv = self.get_pair_indices(ie)
            
            # Get coefficients for quadratic
            a = - self.pppp[iv,io,iv,io]
            b = (2 * (self.F[iv,iv] - self.F[io,io] - 2 * self.pppp[iv,iv,io,io] + self.pppp[io,iv,iv,io]) 
                    + self.pppp[iv,iv,iv,iv] + self.pppp[io,io,io,io])
            c = self.pppp[iv,io,iv,io]
            
            # Solve the quadratic, always picking one root
            t = (- b + np.sqrt(b * b - 4 * a * c)) / (2 * a) 
            # Get coefficients for z equations
            z = - c / (2 * a * t + b)  
            # Add contribution to energy 
            self.Ec += t * self.pppp[iv,io,iv,io]
            # Save t and z
            self.t[io,iv-self.nocc] = t
            self.z[io,iv-self.nocc] = z


    def compute_rdms(self):
        # Compute intermediates
        xji = np.dot(self.z, self.t.T)
        xba = np.dot(self.t.T, self.z) 
        xai = np.multiply(np.power(self.t,2), self.z)

        # Compute 1RDM
        self.dm1 = np.zeros((self.norb,self.norb))
        self.dm1[:self.nocc,:self.nocc] = 2 * np.diag(np.diag(1 - xji)) 
        self.dm1[self.nocc:,self.nocc:] = 2 * np.diag(np.diag(xba))

        # Compute 2RDM
        self.dm2 = np.zeros((self.norb,self.norb,self.norb,self.norb))
        for i in range(self.nocc):
            for j in range(self.nocc):
                # G_{ji,ji} 
                self.dm2[j,i,j,i] = 2 * xji[j,i] 
                if(i==j): self.dm2[j,i,j,i] += 2 * (1 - 2 * xji[i,i])

                # G_{ii,jj}
                self.dm2[i,i,j,j] = 4 * (1 - xji[i,i] - xji[j,j]) 
                if(i==j): self.dm2[i,i,j,j] += 2 * (3 *  xji[i,i] - 1)

            for a in range(self.nocc,2*self.nocc):
                t = self.t[i,a-self.nocc]
                z = self.z[i,a-self.nocc]
                y = xai[i,a-self.nocc]

                # G_{ai,ai}
                self.dm2[a,i,a,i] = 2 * (t + y - 2 * t * (xba[a-self.nocc,a-self.nocc] + xji[i,i] - t * z))
                # G_{ia,ia}
                self.dm2[i,a,i,a] = 2 * z
                # G_{ii,aa}
                self.dm2[i,i,a,a] = 4 * (xba[a-self.nocc,a-self.nocc] - t * z)
                # G_{aa,ii}
                self.dm2[a,a,i,i] = self.dm2[i,i,a,a]

        for a in range(self.nocc, 2*self.nocc):
            for b in range(self.nocc, self.norb): 
                # G_{ba,ba}
                self.dm2[b,a,b,a] = 2 * xba[b-self.nocc,a-self.nocc]
                # G_{aa,bb}
                if(a==b): self.dm2[a,a,b,b] = 2 * xba[a-self.nocc,a-self.nocc]

        for p in range(self.norb):
            for q in range(self.norb):
                if(p!=q):
                    self.dm2[q,p,p,q] = - 0.5 * self.dm2[p,p,q,q]

    @property
    def energy(self):
        """Get the total energy"""
        E = self.E0
        L = self.E0
        for ie in range(self.nocc):
            io, iv = self.get_pair_indices(ie)
            
            # Get coefficients for quadratic
            a = - self.pppp[iv,io,iv,io]
            b = (2 * (self.F[iv,iv] - self.F[io,io] - 2 * self.pppp[iv,iv,io,io] + self.pppp[io,iv,iv,io]) 
                    + self.pppp[iv,iv,iv,iv] + self.pppp[io,io,io,io])
            c = self.pppp[iv,io,iv,io]
            
            # Solve the quadratic, always picking one root
            t = self.t[io,iv-self.nocc]
            z = self.z[io,iv-self.nocc]
            # Add contribution to energy 
            E += t * self.pppp[iv,io,iv,io]
            L += t * self.pppp[iv,io,iv,io] + z * (a * t * t + b * t + c)
        return E # self.E0 + self.Ec

    @property
    def dim(self):
        return self.nrot

    @property
    def gradient(self):
        #F = np.zeros((self.norb,self.norb))
        #F += np.einsum('rp,qr->pq',self.h1e,self.dm1) - np.einsum('qr,rp->pq',self.h1e,self.dm1)
        #F += np.einsum('rpst,qrts->pq',self.pppp, self.dm2) - np.einsum('qrts,rpst->pq', self.pppp, self.dm2)

        dpq = np.identity(self.norb)
        F = np.einsum('mq,nq->mn',self.dm1,self.h1e) + np.einsum('mqrs,nqrs->mn',self.dm2,self.pppp)
        G = 2 * (F - F.T)[self.rot_idx]
        return G

    @property
    def hessian(self):
        dpq = np.identity(self.norb)
        H = np.zeros((self.norb,self.norb,self.norb,self.norb))

        Y  = np.einsum('up,su->sp', self.h1e, self.dm1) + np.einsum('up,su->sp', self.dm1, self.h1e)
        Y += np.einsum('upvt,sutv->sp',self.pppp,self.dm2) + np.einsum('upvt,sutv->sp',self.dm2,self.pppp)

        H += 0.5 * ( np.einsum('qr,sp->pqrs',dpq,Y) + np.einsum('qr,sp->pqrs',Y,dpq) ) 
        H -= (np.einsum('sp,qr->pqrs',self.h1e,self.dm1) + np.einsum('qr,sp->pqrs',self.h1e,self.dm1))
        H += np.einsum('upvr,qusv->pqrs',self.pppp,self.dm2) + np.einsum('qusv,upvr->pqrs',self.pppp,self.dm2)
        H -= (np.einsum('sptu,qrut->pqrs',self.pppp,self.dm2) + np.einsum('tpsu,qtur->pqrs',self.pppp,self.dm2))
        H -= (np.einsum('qrut,sptu->pqrs',self.pppp,self.dm2) + np.einsum('qtur,tpsu->pqrs',self.pppp,self.dm2))

        Ht = np.einsum('pqrs->pqrs',H) - np.einsum('pqsr->pqrs',H)
        H  = np.einsum('pqrs->pqrs',Ht) - np.einsum('qprs->pqrs',Ht)

        hess = (H[:,:,self.rot_idx])[self.rot_idx,:]
        return hess

    def save_last_step(self):
        self.mo_coeff_save = self.mo_coeff.copy()
        return None

    def restore_last_step(self):
        # Restore coefficients
        self.mo_coeff = self.mo_coeff_save.copy()
        # Update integrals
        self.update_integrals()
        return

    def take_step(self,step):
        # Save our last position
        self.save_last_step()
        # Take steps in orbital space
        self.rotate_orb(step)
        # Update integrals
        self.update_integrals()
        # Solve amplitude equations
        self.solve_amplitudes()
        # Compute RDM
        self.compute_rdms()
        return

    def rotate_orb(self,step): 
        orb_step = np.zeros((self.norb,self.norb))
        orb_step[self.rot_idx] = -step
        self.mo_coeff = np.dot(self.mo_coeff, scipy.linalg.expm(orb_step - orb_step.T))
        return

    def overlap(self, other):
        return 0
    def hamiltonian(self, other):
        return 0

    def initialise(self, mo_guess, mat_ci=None, integrals=True):
        # Check orthogonalisation
        self.mo_coeff = orthogonalise(mo_guess, self.ovlp)
        
#        print(self.nocc)
#        print(self.mo_coeff)
##        pm = lo.boys.Boys(self.mol, self.mo_coeff[:,:self.nocc])
#        pm = lo.PM(self.mol, self.mo_coeff[:,:self.nocc])
#        pm.init_guess = 'atomic'
#        Cocc = pm.kernel(verbose=10)
#
#        Cvir = lo.vvo.livvo(self.mol, Cocc, self.mo_coeff[:,self.nocc:], verbose=2)
#        print(Cocc)
#        print(Cvir)
#
#        ov = Cvir.T.dot(self.ovlp).dot(Cocc)
#        print(Cocc[:,[0]].dot(Cvir[:,[0]].T))
#        print(ov)
#        quit()
         


        self.nmo = self.mo_coeff.shape[1]

        # Initialise integrals
        self.update_integrals()

        # Solve the amplitudes
        self.solve_amplitudes()
        # Compute RDM
        self.compute_rdms()

        return None

    def save_to_disk(self, tag):
        # Get the Hessian index
        hindices = self.get_hessian_index()

        # Save coefficients, CI, and energy
        np.savetxt(tag+'.mo_coeff', self.mo_coeff, fmt="% 20.16f")
        np.savetxt(tag+'.amp',      self.t[self.amp_idx], fmt="% 20.16f")
        np.savetxt(tag+'.energy',   
                   np.array([[self.energy, hindices[0], hindices[1], 0]]), 
                   fmt="% 18.12f % 5d % 5d % 12.6f")

    def read_from_disk(self,tag):
        # Read MO coefficient and CI coefficients
        mo_coeff = np.genfromtxt(tag+".mo_coeff")
        # Initialise object
        self.initialise(mo_coeff)
        self.t[self.amp_idx] = np.genfromtxt(tag+'.amp')

    def copy(self):
        new = PP(self.mol)
        new.initialise(self.mo_coeff)
        return new
