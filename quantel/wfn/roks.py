#!/usr/bin/python3
import numpy as np
from quantel.utils.csf_utils import get_ensemble_expansion, get_det_occupation, csf_reorder_orbitals
from quantel.utils.linalg import orthogonalise, stable_eigh, matrix_print
#from quantel.gnme.csf_noci import csf_coupling, csf_coupling_slater_condon
from .csf import CSF
from quantel.utils.orbital_guess import orbital_guess

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
        CSF.__init__(self,integrals,spin_coupling,verbose)


    def initialise(self, mo_guess, spin_coupling=None, mat_ci=None, integrals=True):
        """ Initialise the CSF object with a set of MO coefficients"""
        if(spin_coupling is None):
            spin_coupling = self.spin_coupling
        self.setup_spin_coupling(spin_coupling)
        # We also want to compute the ROKS ensemble coefficients
        self.ensemble_dets = get_ensemble_expansion(spin_coupling)
        self.ensemble_coeff = np.array([c for (_,c) in self.ensemble_dets])

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
    def gradient(self):
        # 1. Get wavefunction part from parent CSF class
        grad = CSF.gradient.fget(self)

        # 2. Compute XC contribution
        grad_xc = np.zeros((self.nmo,self.nmo))
        grad_xc[:self.ncore,:] += np.linalg.multi_dot([self.mo_coeff[:,:self.ncore].T, self.vxc[0], self.mo_coeff])
        for W in range(self.nshell):
            shell = self.shell_indices[W]
            grad_xc[shell,:] += np.linalg.multi_dot([self.mo_coeff[:,shell].T, self.vxc[W+1], self.mo_coeff])

        # 3. Combine and return
        grad += 2 * (grad_xc.T - grad_xc)[self.rot_idx]
        return grad


    @property
    def hessian(self):
        ''' This method finds orb-orb part of the Hessian '''
        # 1. Get wavefunction component from parent CSF class
        hess = CSF.hessian.fget(self)

        # 2.Compute xc_correlation part
        xc_hess = np.zeros((self.nmo,self.nmo,self.nmo,self.nmo))
        # Loop over determinants in the ensemble
        for Idet, (detL, cL) in enumerate(self.ensemble_dets):
            # Get occupation numbers and occupied/virtual orbitals for this determinant
            occ = np.asarray(get_det_occupation(detL, self.shell_indices, self.ncore, self.nmo))

            # Get corresponding vxc in MO basis
            vxc_mo = self.mo_transform(self.vxc_ensemble[Idet])
            # Contribution from xc potential (2nd order orbital and density term)
            for p in range(self.nmo):
                xc_hess[p,:,p,:] += cL * (2 * occ[0][p] - occ[0][:,None] - occ[0][None,:]) * vxc_mo[0]
                xc_hess[p,:,p,:] += cL * (2 * occ[1][p] - occ[1][:,None] - occ[1][None,:]) * vxc_mo[1]

            # Contribution from xc kernel (2nd order functional term)
            if(not (self.integrals.xc is None)):
                # Build ground-state density and xc kernel
                rho0, vxc, fxc = self.integrals.cache_xc_kernel((self.mo_coeff,self.mo_coeff),occ,spin=1)

                # Loop over unique contributions, where s is occupied in alpha or beta space
                for s in np.argwhere(occ[0]+occ[1]>0).flatten():
                    for r in range(self.nmo):
                        # Build the first-order density matrix for this orbital pair
                        D1 = np.einsum('x,m,n->xmn',occ[:,s],self.mo_coeff[:,r],self.mo_coeff[:,s],optimize='optimal')
                        # Compute the contracted kernel with first-order density
                        fxc_ia = self.mo_transform(self.integrals.uks_fxc(D1,rho0,vxc,fxc))
                        # Add contribution
                        xc_hess[r,s] += 4 * cL * np.einsum('xq,xpq->pq',occ,fxc_ia,optimize='optimal')
        # Antisymmetrise
        xc_hess = xc_hess - xc_hess.transpose(1,0,2,3)
        xc_hess = xc_hess - xc_hess.transpose(0,1,3,2)

        # 3. Combine wavefunction with xc and return
        hess += (xc_hess[:,:,self.rot_idx])[self.rot_idx,:]
        return hess


    def hess_on_vec(self, vec):
        """ Compute the Hessian @ vec product directly, without forming the full Hessian matrix 
            This is more memory efficient and avoids any ERI computation.

            Inputs:
                vec : vector to be multiplied by the Hessian
            Returns:
                Hvec : result of Hessian @ vec product
        """
        # 0. Get antisymmetric step from vector
        step = np.zeros((self.nmo,self.nmo))
        step[self.rot_idx] = vec
        step -= step.T

        # 1. Get wavefunction part from parent CSF class (already formatted)
        Hvec = CSF.hess_on_vec(self,vec)

        # 2. Compute xc contribution
        xc_Hvec = np.zeros_like(step)
        # Loop over determinants in the ensemble
        for Idet, (detL, cL) in enumerate(self.ensemble_dets):
            # Get occupation numbers and occupied/virtual orbitals for this determinant
            occ = np.asarray(get_det_occupation(detL, self.shell_indices, self.ncore, self.nmo))
            
            # Get corresponding vxc in MO basis
            vxc_mo = self.mo_transform(self.vxc_ensemble[Idet])
            # Contribution from xc potential (2nd order orbital and density term)
            # here x is an index for the spin channels
            xc_Hvec += cL * (2 * np.einsum('xp,ps,xsq->pq',occ,step,vxc_mo,optimize='optimal') 
                               - np.einsum('ps,xs,xsq->pq',step,occ,vxc_mo,optimize='optimal')
                               - np.einsum('ps,xsq,xq->pq',step,vxc_mo,occ,optimize='optimal'))

            # Contribution from xc kernel (2nd order functional term)
            if(not (self.integrals.xc is None)):
                # Build ground-state density and xc kernel
                rho0, vxc, fxc = self.integrals.cache_xc_kernel((self.mo_coeff,self.mo_coeff),occ,spin=1)
                # Build the first-order density matrix for this orbital pair
                D1 = np.einsum('xs,mr,rs,ns->xmn',occ,self.mo_coeff,step,self.mo_coeff,optimize='optimal')
                # Compute the contracted kernel with first-order density
                fxc_ia = self.mo_transform(self.integrals.uks_fxc(D1,rho0,vxc,fxc))
                # Add contribution (here x is an index for the spin channels)
                xc_Hvec += 4 * cL * np.einsum('xq,xpq->pq',occ,fxc_ia,optimize='optimal')
        # Antisymmetrise
        xc_Hvec = xc_Hvec - xc_Hvec.T

        ## 3. Combine wavefunction with xc and return
        Hvec += xc_Hvec[self.rot_idx]
        return Hvec
    

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
            matrix_print(self.gen_fock_xc[:self.nocc,:].T, title="Generalised Fock Matrix (MO basis)")
        print()


    def get_vxc(self):
        """ Compute xc-potential from sum of contributions from each determinant in the ensemble
            Returns:
                exc      : exchange-correlation energy
                vxc_shell: list of xc-potential contributions per shell
        """
        # Initialise overall variables
        exc = 0.0
        vxc_shell = np.zeros((self.nshell+1,self.nbsf,self.nbsf))
        if(self.nopen == 0):
            # Restricted case
            rho = self.dk[0]
            exc, vxc = self.integrals.build_vxc(rho,hermi=1)
            vxc_shell[0] = vxc[0] + vxc[1]
            vxc_ensemble = vxc_shell[0]
        else:
            # loop over determinants in the ensemble
            vxc_ensemble = np.zeros((len(self.ensemble_dets),2,self.nbsf,self.nbsf))
            for Idet, (det_str, coeff) in enumerate(self.ensemble_dets):
                # Initialise spin densities from core contribution
                dma, dmb = 0.5 * self.dk[0], 0.5 * self.dk[0]

                # Add open-shell contributions depending on spin occupation of this determinant
                for Ishell, spinI in enumerate(det_str):
                    if(spinI == 'a'): dma += self.dk[1+Ishell]
                    else: dmb += self.dk[1+Ishell]

                # Build the vxc for this determinant
                exc_det, vxc_det = self.integrals.build_vxc((dma,dmb),hermi=1)
                vxc_ensemble[Idet] = vxc_det
                # Accumulate the energy
                exc += coeff * exc_det

                # Core contribution
                if(self.ncore > 0):
                    vxc_shell[0] += coeff * (vxc_det[0] + vxc_det[1])
                # Open-shell contributions
                for Ishell, spinI in enumerate(det_str):
                    vxc_shell[1+Ishell] += coeff * (vxc_det[0] if (spinI=='a') else vxc_det[1])
                
        return exc, vxc_shell, vxc_ensemble


    def update_integrals(self):
        """ Update the integrals with current set of orbital coefficients"""
        # Update density, J, K, wfn_fock and gen_fock from parent CSF class
        CSF.update_integrals(self)
        # Compute xc-potential
        self.exc, self.vxc, self.vxc_ensemble = self.get_vxc()
        # Get DFT generalised Fock matrix
        self.gen_fock_xc = self.gen_fock.copy()
        self.gen_fock_xc[:self.ncore,:] += np.linalg.multi_dot([self.mo_coeff[:,:self.ncore].T, self.vxc[0], self.mo_coeff])
        for W, shell in enumerate(self.shell_indices):
            self.gen_fock_xc[shell,:] += np.linalg.multi_dot([self.mo_coeff[:,shell].T, self.vxc[W+1], self.mo_coeff])
        # Add XC contribution to overall Fock matrix
        self.fock += np.einsum('Lxpq,L->pq',self.vxc_ensemble,self.ensemble_coeff)


    def copy(self,integrals=True):
        """Return a copy of the current object"""
        newcsf = ROKS(self.integrals, self.spin_coupling, verbose=self.verbose)
        newcsf.initialise(self.mo_coeff,spin_coupling=self.spin_coupling,integrals=integrals)
        return newcsf


    def overlap(self, them):
        """ Compute the overlap between two CSF objects
        """
        raise NotImplementedError("ROKS overlap coupling not yet implemented")


    def hamiltonian(self, them):
        """ Compute the Hamiltonian coupling between two CSF objects
        """
        raise NotImplementedError("ROKS Hamiltonian coupling not yet implemented")
    

    def get_orbital_guess(self, method="gwh",avas_ao_labels=None,reorder=True):
        """Get a guess for the molecular orbital coefficients"""
        # Get the guess for the molecular orbital coefficients
        Cguess = orbital_guess(self.integrals,method,avas_ao_labels=avas_ao_labels,rohf_ms=0.5*self.nopen)
        # Optimise the order of the CSF orbitals and return
        if(reorder and (self.spin_coupling != '')):
            Cguess[:,self.ncore:self.nocc] = csf_reorder_orbitals(self.integrals,self.exchange_matrix,
                                                                  np.copy(Cguess[:,self.ncore:self.nocc]))

        # Initialise the CSF object with the guess coefficients.
        self.initialise(Cguess, spin_coupling=self.spin_coupling)
        return


    def get_preconditioner(self,abs=True,include_fxc=False):
        """Compute approximate diagonal of Hessian"""
         # Initialise with diagonal from wfn part
        Q = CSF.get_preconditioner(self,abs=False)

        # Compute xc contribution
        Q_xc = np.zeros((self.nmo,self.nmo))
        for Idet, (detL, cL) in enumerate(self.ensemble_dets):
            # Get occupation numbers and occupied/virtual orbitals for this determinant
            occ = np.asarray(get_det_occupation(detL, self.shell_indices, self.ncore, self.nmo))            
            # Get corresponding vxc in MO basis
            vxc_mo = self.mo_transform(self.vxc_ensemble[Idet])
            diag_vxc_mo = np.einsum('xpp->xp',vxc_mo)
            # Contribution from xc potential (2nd order orbital and density term)
            Q_xc += 2 * cL * np.einsum('xpq,xpq->pq',occ[:,:,None]-occ[:,None,:], diag_vxc_mo[:,None,:]-diag_vxc_mo[:,:,None])

            # Contribution from xc kernel (2nd order functional term).
            # This is an expensive computation, so only include if requested (see include_fxc flag)
            if((not (self.integrals.xc is None)) and include_fxc):
                # Build ground-state density and xc kernel
                rho0, vxc, fxc = self.integrals.cache_xc_kernel((self.mo_coeff,self.mo_coeff),occ,spin=1)
                # Loop over unique contributions, where s is occupied in alpha or beta space
                for p in range(self.nmo):
                    for q in range(p):
                        # Build the first-order density matrix for this orbital pair
                        D1 = np.outer(self.mo_coeff[:,p],self.mo_coeff[:,q])
                        D1a = (occ[0][q] - occ[0][p]) * D1
                        D1b = (occ[1][q] - occ[1][p]) * D1
                        # Compute the contracted kernel with first-order density
                        fxc_ia = self.mo_transform(self.integrals.uks_fxc([D1a,D1b],rho0,vxc,fxc))
                        # Add contribution
                        Q_xc[p,q] += 4 * cL * (occ[0][q] - occ[0][p]) * fxc_ia[0][p,q]
                        Q_xc[p,q] += 4 * cL * (occ[1][q] - occ[1][p]) * fxc_ia[1][p,q]

        # Combine wfn and xc, and return
        Q += Q_xc[self.rot_idx]
        return np.abs(Q) if abs else Q
    

    def koopmans(self):
        """ Solve IP using Extended Koopmans theory"""
        from scipy.linalg import eigh
        # Transform gen Fock matrix to MO basis
        e, v = eigh(-self.gen_fock_xc[:self.nocc,:self.nocc], np.diag(self.mo_occ[:self.nocc]))
        # Normalize ionization orbitals wrt standard metric
        for i in range(self.nocc):
            v[:,i] /= np.linalg.norm(v[:,i])
        # Convert ionization orbitals to MO basis
        cip = self.mo_coeff[:,:self.nocc].dot(v)
        # Compute occupation of ionization orbitals
        occip = np.diag(np.einsum('ip,i,iq->pq',v,self.mo_occ[:self.nocc],v))   
        return e, cip, occip


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
        foo = self.gen_fock_xc[self.core_indices,:][:,self.core_indices]
        self.mo_energy[:self.ncore], Qoo = stable_eigh(foo)
        for i, ii in enumerate(self.core_indices):
            for j, jj in enumerate(self.core_indices):
                Q[ii,jj] = Qoo[i,j]
        # Scale core orbital energies
        self.mo_energy[:self.ncore] *= 0.5

        # Loop over shells
        for W in self.shell_indices:
            fww = self.gen_fock_xc[W,:][:,W]
            self.mo_energy[W], Qww = stable_eigh(fww)
            for i, ii in enumerate(W):
                for j, jj in enumerate(W):
                    Q[ii,jj] = Qww[i,j]

        # Virtual transformation. Here we use the standard Fock matrix
        fvv = np.linalg.multi_dot([self.mo_coeff[:,self.nocc:].T, self.fock, self.mo_coeff[:,self.nocc:]])
        self.mo_energy[self.nocc:], Qvv = stable_eigh(fvv)
        Q[self.nocc:,self.nocc:] = Qvv

        # Apply transformation
        if(np.linalg.det(Q) < 0): Q[:,0] *= -1
        self.mo_coeff = self.mo_coeff @ Q
        
        # Update integrals
        self.update_integrals()
        return Q