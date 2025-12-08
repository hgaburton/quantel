#!/usr/bin/python3
# Modified from ss_casscf code of Antoine Marie and Hugh G. A. Burton
# This is code for a CSF, which can be formed in a variety of ways.
import numpy as np
import scipy, quantel, h5py
from quantel.utils.csf_utils import get_csf_vector, get_ensemble_expansion
from quantel.utils.linalg import orthogonalise, stable_eigh, matrix_print
from quantel.gnme.csf_noci import csf_coupling, csf_coupling_slater_condon
from .csf import CSF
from quantel.utils.csf_utils import csf_reorder_orbitals
from quantel.utils.orbital_guess import orbital_guess
from quantel.ints.pyscf_integrals import PySCFIntegrals

class ROKS(CSF):
    """ 
        A class for a ROKS using arbitrary genealogical coupling pattern.

        Inherits from the Wavefunction abstract base class with pure virtual properties:
            - energy
            - gradient
            - hessian
            - take_step
            - save_last_step
            - restore_step
    """
    def __init__(self, integrals, spin_coupling, verbose=0):
        """ Initialise the CSF wave function
                integrals     : quantel integral interface
                spin_coupling : genealogical coupling pattern
                verbose       : verbosity level
        """
        # Call the parent constructor
        super().__init__(integrals, spin_coupling, verbose)
        # We also want to compute the ROKS ensemble coefficients
        self.ensemble_dets = get_ensemble_expansion(spin_coupling)
        # And store information about our XC functional

    def initialise(self, mo_guess, spin_coupling=None, mat_ci=None, integrals=True):
        """ Initialise the CSF object with a set of MO coefficients"""
        if(spin_coupling is None):
            spin_coupling = self.spin_coupling
        self.setup_spin_coupling(spin_coupling)

        # Orthogonalise the MO coefficients
        mo_guess      = orthogonalise(mo_guess, self.integrals.overlap_matrix())        
        if(mo_guess.shape[1] != self.nmo):
            raise ValueError("Number of orbitals in MO coefficient matrix is incorrect")
        self.mo_coeff = mo_guess

        # Save mapping indices for unique orbital rotations
        self.frozen     = None
        self.rot_idx    = self.uniq_var_indices(self.frozen)
        self.invariant  = self.invariant_indices()
        self.nrot       = np.sum(self.rot_idx)

        # Initialise integrals
        if (integrals): self.update_integrals()

    @property
    def energy(self):
        """ Compute the energy corresponding to a given set of
             one-el integrals, two-el integrals, 1- and 2-RDM
        """
        # Nuclear repulsion
        En = self.integrals.scalar_potential()
        # One-electron energy
        E1 = np.einsum('pq,qp',self.dj,self.integrals.oei_matrix(True))
        # Coulomb energy
        EJ = 0.5 * np.einsum('pq,qp',self.dj,self.J)
        # Exchange energy
        EK = - 0.25 * np.einsum('pq,qp',self.dj,self.K[0])
        for w in range(self.nshell):
            EK += 0.5 * np.einsum('pq,qp',self.K[1+w], 
                        np.einsum('v,vpq->pq',self.beta[w],self.dk[1:]) - 0.5 * self.dk[0])
        # xc-potential energy
        Exc = self.exc
        # Save components
        self.energy_components = dict(Nuclear=En, One_Electron=E1, Coulomb=EJ, 
                                      ROHF_Exchange=EK, Exchange_Correlation=Exc)
        return En + E1 + EJ + EK + Exc

    @property
    def exchange_matrix(self):
        """ Compute the exchange matrix in the MO basis"""
        Xb = np.zeros((self.nocc,self.nocc))
        # Open-Open
        for W, sW in enumerate(self.shell_indices):
            for V, sV in enumerate(self.shell_indices):
                for w in sW:
                    for v in sV:
                        Xb[w,v] = self.beta[W,V]
        # Set diagonal to zero
        np.fill_diagonal(Xb,0)
        return Xb[self.ncore:,self.ncore:]

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
        if(verbose > 0):
            print("\n ---------------------------------------------")
            print(f"         Total Energy = {self.energy:14.8f} Eh")
            for key, value in self.energy_components.items():
                print(f" {key.replace('_',' '):>20s} = {value:14.8f} Eh")
            print(" ---------------------------------------------")
            print(f"        <Sz> = {self.sz:5.2f}")
            print(f"        <S2> = {self.s2:5.2f}")
        if(verbose > 1):
            matrix_print(self.exchange_matrix, title="Open-Shell Exchange Matrix <Ψ|Êpq Êqp|Ψ> - Np")
            matrix_print(self.Ipqqp[:self.nocc,self.ncore:self.nocc], title="Open-Shell Exchange Integrals <pq|qp>")
        if(verbose > 2):
            matrix_print(self.mo_coeff[:,:self.nocc], title="Occupied Orbital Coefficients")
        if(verbose > 3):
            matrix_print(self.mo_coeff[:,self.nocc:], title="Virtual Orbital Coefficients", offset=self.nocc)
        if(verbose > 4):
            matrix_print(self.gen_fock[:self.nocc,:].T, title="Generalised Fock Matrix (MO basis)")
        print()

    def approx_hess_on_vec(self, vec, eps=1e-3):
        """ Compute the approximate Hess * vec product using forward finite difference """
        # Get current gradient
        g0 = self.gradient.copy()

        # Get forward gradient
        # First copy CSF but don't initialise integrals
        them = self.copy(integrals=False)
        # Take step (this will evaluate necessary integrals)
        them.take_step(eps * vec)
        g1 = them.gradient.copy()
        # Parallel transport back to current position
        g1 = self.transform_vector(g1, - eps * vec)
        
        # Return approximation to H @ sk
        return (g1 - g0) / eps


    def hess_on_vec(self, vec):
        return self.hessian @ vec


    def get_vxc(self):
        """Compute xc-potential"""
        # Initialise overall variables
        exc = 0.0
        vxc_shell = np.zeros((self.nshell+1,self.nbsf,self.nbsf))

        if(self.nopen == 0):
            # Restricted case
            rho = self.dk[0]
            exc, vxc = self.integrals.build_vxc(rho)
            vxc_shell[0] = vxc[0] + vxc[1]
        else:
            # loop over determinants in the ensemble
            for det_str, coeff in self.ensemble_dets:
                # Initialise spin densities from core contribution
                dma = 0.5 * self.dk[0]
                dmb = 0.5 * self.dk[0]

                # Add open-shell contributions depending on spin occupation of this determinant
                for Ishell, spinI in enumerate(det_str):
                    dshell = self.dk[1+Ishell]
                    if(spinI == 'a'): dma += dshell
                    else: dmb += dshell

                # Build the vxc for this determinant
                exc_det, (vxca_det,vxcb_det) = self.integrals.build_vxc((dma,dmb))
                # Accumulate the energy
                exc += coeff * exc_det

                # Core contribution
                if(self.ncore > 0):
                    vxc_shell[0] += coeff * (vxca_det + vxcb_det)
                # Open-shell contributions
                for Ishell, spinI in enumerate(det_str):
                    vxc_shell[1+Ishell] += coeff * (vxca_det if (spinI=='a') else vxcb_det)

        return exc, vxc_shell


    def update_integrals(self):
        """ Update the integrals with current set of orbital coefficients"""
        # Update density matrices (AO basis)
        self.dj, self.dk, self.vd = self.get_density_matrices()
        # Compute xc-potential
        self.exc, self.vxc = self.get_vxc()
        # Update JK matrices (AO basis) 
        self.J, self.K = self.get_JK_matrices(self.vd)
        # Get Fock matrix (AO basis)
        self.fock = self.integrals.oei_matrix(True) + self.J - 0.5 * np.einsum('mpq->pq',self.K)
        # Get generalized Fock matrices
        self.gen_fock, self.Ipqpq, self.Ipqqp = self.get_generalised_fock()
        return 


    def write_fcidump(self, tag):
        """ Write an FCIDUMP file for the current CSF object """
        if(not (type(self.integrals) is quantel.ints.pyscf_integrals.PySCFIntegrals)):
            raise ValueError("FCIDUMP file can only be written for PySCF integrals")
        
        # Write the FCIDUMP using PySCF
        from pyscf.tools import fcidump
        mol = self.integrals.molecule().copy()
        mol.spin = int(2 * self.sz)
        fcidump.from_mo(mol, tag+'.fcid', self.mo_coeff, ms=self.sz)

    def write_cidump(self, tag):
        # Write the CI vector dump
        from quantel.utils.ci_utils import write_cidump
        write_cidump(get_csf_vector(self.spin_coupling),self.ncore,self.nbsf,tag+'_civec.txt')


    def copy(self,integrals=True):
        """Return a copy of the current object"""
        newcsf = CSF(self.integrals, self.spin_coupling, verbose=self.verbose)
        newcsf.initialise(self.mo_coeff,spin_coupling=self.spin_coupling,integrals=integrals)
        return newcsf


    def overlap(self, them):
        """ Compute the overlap between two CSF objects
        """
        ovlp = self.integrals.overlap_matrix()
        return csf_coupling(self, them, ovlp)[0]


    def hamiltonian(self, them):
        """ Compute the Hamiltonian coupling between two CSF objects
        """
        return csf_coupling_slater_condon(self, them, self.integrals)
    

    def get_orbital_guess(self, method="gwh",avas_ao_labels=None,reorder=True):
        """Get a guess for the molecular orbital coefficients"""
        # Get the guess for the molecular orbital coefficients
        Cguess = orbital_guess(self.integrals,method,avas_ao_labels=avas_ao_labels,rohf_ms=0.5*self.nopen)
        # Optimise the order of the CSF orbitals and return
        if(reorder and (not self.spin_coupling is '')):
            Cguess[:,self.ncore:self.nocc] = csf_reorder_orbitals(self.integrals,self.exchange_matrix,
                                                                  np.copy(Cguess[:,self.ncore:self.nocc]))

        # Initialise the CSF object with the guess coefficients.
        self.initialise(Cguess, spin_coupling=self.spin_coupling)
        return
    

    def get_spin_density(self):
        """ Compute the alfa and beta density matrices"""
        dm_tmp = np.einsum('kpq->pq',self.dk[1:])
        rho_a, rho_b = 0.5 * self.dk[0], 0.5 * self.dk[0]
        if(self.nopen > 0):
            rho_a += (0.5 + self.sz / self.nopen) * dm_tmp
            rho_b += (0.5 - self.sz / self.nopen) * dm_tmp
        return rho_a, rho_b


    def get_generalised_fock(self):
        """ Compute the generalised Fock matrix in MO basis"""
        # Initialise memory
        F = np.zeros((self.nmo, self.nmo)) 

        # Memory for diagonal elements
        self.gen_fock_diag = np.zeros((self.nmo,self.nmo))
        # Core contribution
        Fcore_ao = 2 * (self.integrals.oei_matrix(True) + self.J 
                      - 0.5 * np.sum(self.K[i] for i in range(self.nshell+1)))
        # XC potential contribution
        Fcore_ao += self.vxc[0]
        # AO-to-MO transformation
        Fcore_mo = np.linalg.multi_dot([self.mo_coeff.T, Fcore_ao, self.mo_coeff])
        for i in range(self.ncore):
            self.gen_fock_diag[i,:] = Fcore_mo.diagonal()
        F[:self.ncore,:] = Fcore_mo[:self.ncore,:]

        # Open-shell contributions
        for W in range(self.nshell):
            # Get shell indices and coefficients
            shell = self.shell_indices[W]
            # One-electron matrix, Coulomb and core exchange
            Fw_ao = self.integrals.oei_matrix(True) + self.J - 0.5 * self.K[0]
            # Different shell exchange
            Fw_ao += np.einsum('v,vpq->pq',self.beta[W],self.K[1:])
            # XC potential contribution
            Fw_ao += self.vxc[W+1]
            # AO-to-MO transformation
            Fw_mo = np.linalg.multi_dot([self.mo_coeff.T, Fw_ao, self.mo_coeff])
            for w in shell:
                self.gen_fock_diag[w,:] = Fw_mo.diagonal()
            F[shell,:] = Fw_mo[shell,:]
        
        # Get diagonal J/K terms
        Ipqpq = np.zeros((self.nopen,self.nmo))
        Ipqqp = np.zeros((self.nopen,self.nmo))
        for i in range(self.nopen):
            Ji = self.mo_coeff.T @ self.vJ[1+i] @ self.mo_coeff
            Ki = self.mo_coeff.T @ self.vIpqqp[1+i] @ self.mo_coeff
            Ipqpq[i] = np.diag(Ji)
            Ipqqp[i] = np.diag(Ki)
        return F, Ipqpq, Ipqqp


    def get_Y_intermediate(self):
        """ Compute the Y intermediate required for Hessian evaluation
        """
        # Get required constants
        nmo   = self.nmo
        ncore = self.ncore
        nocc = self.nocc

        # Get required two-electron MO integrals
        Cocc = self.mo_coeff[:,:nocc].copy()
        ppoo = self.integrals.tei_ao_to_mo(self.mo_coeff,self.mo_coeff,Cocc,Cocc,True,False)
        popo = self.integrals.tei_ao_to_mo(self.mo_coeff,Cocc,self.mo_coeff,Cocc,True,False)

        # K and J in MO basis
        Jmn  = np.einsum('pm,pq,qn->mn',self.mo_coeff, self.J, self.mo_coeff)
        vKmn = np.einsum('pm,wpq,qn->wmn',self.mo_coeff, self.K, self.mo_coeff)
        Kmn  = np.einsum('wpq->pq', vKmn)

        # Build Ypqrs
        Y = np.zeros((nmo,nmo,nmo,nmo))
        print(self.Kscale)
        self.Kscale = 0.0
        # Y_imjn
        Y[:ncore,:,:ncore,:] += 8 * np.einsum('mnij->imjn',ppoo[:,:,:ncore,:ncore]) 
        Y[:ncore,:,:ncore,:] -= 2 * self.Kscale * np.einsum('mnji->imjn',ppoo[:,:,:ncore,:ncore])
        Y[:ncore,:,:ncore,:] -= 2 * self.Kscale * np.einsum('mjni->imjn',popo[:,:ncore,:,:ncore])
        for i in range(ncore):
            Y[i,:,i,:] += 2 * Jmn - Kmn

        # Y_imwn
        Y[:ncore,:,ncore:nocc,:] = (4 * ppoo[:,:,:ncore,ncore:nocc].transpose(2,0,3,1)
                                      - self.Kscale * ppoo[:,:,ncore:nocc,:ncore].transpose(3,0,2,1)
                                      - self.Kscale * popo[:,ncore:nocc,:,:ncore].transpose(3,0,1,2))
        Y[ncore:nocc,:,:ncore,:] = Y[:ncore,:,ncore:nocc,:].transpose(2,3,0,1)

        # Y_wmvn
        for W in range(self.nshell):
            wKmn = np.einsum('v,vmn->mn',self.beta[W], vKmn[1:])
            for V in range(W,self.nshell):
                for w in self.shell_indices[W]:
                    for v in self.shell_indices[V]:
                        Y[w,:,v,:] = 2 * ppoo[:,:,w,v] + self.Kscale * self.beta[W,V] * (ppoo[:,:,v,w] + popo[:,v,:,w])
                        if(w==v):
                            Y[w,:,w,:] = Y[w,:,w,:] + Jmn - 0.5 * vKmn[0] + wKmn
                        else:
                            Y[v,:,w,:] = Y[w,:,v,:].T
        return Y

    def get_preconditioner(self):
        """Compute approximate diagonal of Hessian"""
        # Initialise approximate preconditioner
        Q = np.zeros((self.nmo,self.nmo))

        # Include dominate generalised Fock matrix terms
        for p in range(self.nmo):
            for q in range(p):
                Q[p,q] = 2 * ( (self.gen_fock_diag[p,q] - self.gen_fock_diag[q,q]) 
                             + (self.gen_fock_diag[q,p] - self.gen_fock_diag[p,p]) )

        # Compute two-electron corrections involving active orbitals
        Acoeff = self.Ipqqp
        for q in range(self.ncore,self.nocc):
            for p in range(q):
                Q[q,p] += 4 * (self.mo_occ[p]-self.mo_occ[q])**2 * Acoeff[q-self.ncore,p]
            for p in range(q+1,self.nmo):
                Q[p,q] += 4 * (self.mo_occ[p]-self.mo_occ[q])**2 * Acoeff[q-self.ncore,p]

        # Compute two-electron hybrid exchange contribution
        Bcoeff = self.integrals.hybrid_K * (self.Ipqpq + self.Ipqqp)
        for W in range(self.nshell): 
            for q in self.shell_indices[W]:
                # Core-Active
                for p in range(self.ncore):
                    Q[q,p] -= 2 * Bcoeff[q-self.ncore,p]
                # Active-Active
                for V in range(W):
                    for p in self.shell_indices[V]:
                        Q[q,p] -= 4 * (1 + self.beta[V,W]) * Bcoeff[q-self.ncore,p]
                # Virtual-Active
                for p in range(self.nocc,self.nmo):
                    Q[p,q] -= 2 * (self.mo_occ[p] + self.mo_occ[q]) * Bcoeff[q-self.ncore,p]

        return Q[self.rot_idx]
        return np.abs(Q[self.rot_idx])

    def koopmans(self):
        """
        Solve IP using Koopmans theory
        """
        from scipy.linalg import eigh
        # Transform gen Fock matrix to MO basis
        gen_fock = self.gen_fock[:self.nocc,:self.nocc]
        gen_dens = np.diag(self.mo_occ[:self.nocc])
        e, v = eigh(-gen_fock, gen_dens)
        # Normalize ionization orbitals wrt standard metric
        for i in range(self.nocc):
            v[:,i] /= np.linalg.norm(v[:,i])
        # Convert ionization orbitals to MO basis
        cip = self.mo_coeff[:,:self.nocc].dot(v)
        # Compute occupation of ionization orbitals
        occip = np.diag(np.einsum('ip,i,iq->pq',v,self.mo_occ[:self.nocc],v))   
        return e, cip, occip

    def get_active_itegrals(self):
        """
        Get active space one- and two-electron integrals in MO basis
        """
        Cact = self.mo_coeff[:,self.ncore:self.nocc]
        h1e_act = np.linalg.multi_dot([Cact.T, self.integrals.oei_matrix(True), Cact])
        tei_act = self.integrals.tei_ao_to_mo(Cact,Cact,Cact,Cact,True,False)
        return h1e_act, tei_act

    def canonicalize(self):
        """
        Forms the canonicalised MO coefficients by diagonalising invariant 
        subblocks of the generalised Fock matrix.
        
        For the virtual orbitals, we must use the standard Fock matrix as generalised Fock 
        matrix is all zero.
        """
        # Initialise transformation matrix
        self.mo_energy = np.zeros(self.nmo)
        Q = np.zeros((self.nmo,self.nmo))

        # Get core transformation using generalised Fock matrix
        foo = self.gen_fock[self.core_indices,:][:,self.core_indices]
        self.mo_energy[:self.ncore], Qoo = stable_eigh(foo)
        for i, ii in enumerate(self.core_indices):
            for j, jj in enumerate(self.core_indices):
                Q[ii,jj] = Qoo[i,j]

        # Loop over shells
        for W in self.shell_indices:
            fww = self.gen_fock[W,:][:,W]
            self.mo_energy[W], Qww = stable_eigh(fww)
            for i, ii in enumerate(W):
                for j, jj in enumerate(W):
                    Q[ii,jj] = Qww[i,j]

        # Virtual transformation
        # Here we use the standard Fock matrix
        fvv = np.linalg.multi_dot([self.mo_coeff[:,self.nocc:].T, self.fock, self.mo_coeff[:,self.nocc:]])
        self.mo_energy[self.nocc:], Qvv = stable_eigh(fvv)
        Q[self.nocc:,self.nocc:] = Qvv

        # Apply transformation
        if(np.linalg.det(Q) < 0): Q[:,0] *= -1
        self.mo_coeff = self.mo_coeff @ Q
        
        # Update generalised Fock matrix and diagonal approximations
        self.gen_fock, self.Ipqpq, self.Ipqqp = self.get_generalised_fock()
        return Q
