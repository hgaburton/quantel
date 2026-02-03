#!/usr/bin/python3
# Modified from ss_casscf code of Antoine Marie and Hugh G. A. Burton
# This is code for a CSF, which can be formed in a variety of ways.
from pdb import pm
import numpy as np
import scipy, quantel, h5py
from quantel.utils.csf_utils import verify_spin_coupling, get_shells, get_shell_exchange, get_csf_vector
from quantel.utils.linalg import orthogonalise, stable_eigh, matrix_print
from quantel.gnme.csf_noci import csf_coupling, csf_coupling_slater_condon
from .wavefunction import Wavefunction
from quantel.utils.csf_utils import csf_reorder_orbitals
from quantel.utils.orbital_guess import orbital_guess
from quantel.utils.orbital_utils import localise_orbitals

def flag_transport(A,T,mask,max_order=50,tol=1e-4):
   tA = A.copy()
   M  = A.copy()
   for i in range(max_order):
       TM = T @ M
       M = - 0.5 * (TM - TM.T) / (i+1)
       M[mask] = 0
       if(np.max(np.abs(M)) < tol):
           break
       tA += M
   return tA

class CSF(Wavefunction):
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
    def __init__(self, integrals, spin_coupling, verbose=0, advanced_preconditioner=False):
        """ Initialise the CSF wave function
                integrals     : quantel integral interface
                spin_coupling : genealogical coupling pattern
                verbose       : verbosity level
        """
        if(type(self)==CSF and (integrals.xc != None)):
            raise ValueError("CSF class cannot be used with DFT functionals - use ROKS class instead")

        # How noisy am I?
        self.verbose       = verbose
        # Initialise integrals object
        self.integrals  = integrals
        self.nalfa      = integrals.molecule().nalfa()
        self.nbeta      = integrals.molecule().nbeta()
        # Get number of basis functions and linearly independent orbitals
        self.nbsf       = integrals.nbsf()
        self.nmo        = integrals.nmo()
        # Control if the open-shell J/K integrals are used for preconditioner (scaling Nopen * N^4)
        # Default is not to use these integrals as they incur more JK builds than the gradient
        self.advanced_preconditioner = advanced_preconditioner
        # Initialise spin coupling 
        self.setup_spin_coupling(spin_coupling)
    
    def sanity_check(self):
        '''Need to be run at the start of the kernel to verify that the number of 
           orbitals and electrons in the CAS are consistent with the system '''
        # Check number of active orbitals is positive
        if self.nopen < 0:
            raise ValueError("Number of active orbitals must be positive")
        # Check number of active electrons doesn't exceed total number of electrons
        if(self.cas_nalfa > self.nopen or self.cas_nbeta > self.nopen):
            raise ValueError("Number of active electrons must be <= number of active orbitals")
        # Check number of active electrons doesn't exceed total number of electrons
        if(self.cas_nalfa > self.nalfa or self.cas_nbeta > self.nbeta):
            raise ValueError("Number of active electrons must be <= total number of electrons")
        # Check number of occupied orbitals doesn't exceed total number of orbitals
        if(self.nocc > self.nmo):
            raise ValueError("Number of inactive and active orbitals must be <= total number of orbitals")

    def setup_spin_coupling(self, spin_coupling): 
        """ Setup the spin coupling pattern for the CSF wave function """
        if(spin_coupling == 'cs'):
            spin_coupling = ''

        # Verify
        verify_spin_coupling(spin_coupling)
        # Get active space definition
        self.nopen   = len(spin_coupling)
        self.cas_nalfa  = sum(int(s=='+') for s in spin_coupling)
        self.cas_nbeta  = sum(int(s=='-') for s in spin_coupling)
        # Get number of core electrons
        self.ncore = self.integrals.molecule().nalfa() + self.integrals.molecule().nbeta() - self.cas_nalfa - self.cas_nbeta
        if(self.ncore % 2 != 0):
            raise ValueError("Number of core electrons must be even")
        if(self.ncore < 0):
            raise ValueError("Number of core electrons must be positive")
        self.ncore = self.ncore // 2
        self.nalfa = self.ncore + self.cas_nalfa
        self.nbeta = self.ncore + self.cas_nbeta
        # Get numer of 'occupied' orbitals
        self.nocc = self.ncore + self.nopen
        self.sanity_check()

        # Get determinant list and CSF occupation/coupling vectors
        self.spin_coupling = spin_coupling
        self.core_indices, self.shell_indices = get_shells(self.ncore,self.spin_coupling)
        self.mo_occ = np.zeros(self.nmo)
        self.mo_occ[:self.nocc] = 2
        self.mo_occ[self.ncore:self.nocc] = 1
        # Get information about the electron shells
        self.beta   = get_shell_exchange(self.ncore,self.shell_indices, self.spin_coupling)
        self.nshell = len(self.shell_indices)
        # Get anion exchange for canonicalisation
        self.anion_dbeta = self.get_anion_exchange()

    def initialise(self, mo_guess, spin_coupling=None, mat_ci=None, integrals=True):
        """ Initialise the CSF object with a set of MO coefficients"""
        if(spin_coupling == None):
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
        if (integrals): self.update()

    def deallocate(self):
        pass

    @property
    def dim(self):
        """Number of degrees of freedom"""
        return self.nrot

    @property
    def energy(self):
        """ Compute the energy corresponding to a given set of
             one-el integrals, two-el integrals, 1- and 2-RDM
        """
        # Total density
        dt = np.einsum('wpq->pq',self.vd)
        # Nuclear repulsion
        En = self.integrals.scalar_potential()
        # One-electron energy
        E1 = np.einsum('pq,qp',dt,self.integrals.oei_matrix(True))
        # Coulomb energy
        EJ = 0.5 * np.einsum('pq,qp',dt,self.J)
        # Exchange energy
        EK = - 0.25 * np.einsum('pq,qp',dt,self.K[0])
        for w in range(self.nshell):
            EK += 0.5 * np.einsum('pq,qp',self.K[1+w], 
                        np.einsum('v,vpq->pq',self.beta[w],self.vd[1:]) - 0.5 * self.vd[0])
        # Save components
        self.energy_components = dict(Nuclear=En, One_Electron=E1, Coulomb=EJ, ROHF_Exchange=EK)
        return En + E1 + EJ + EK
    
    @property
    def sz(self):
        """<S_z> value of the current wave function"""
        return 0.5 * np.sum([1 if s=='+' else -1 for s in self.spin_coupling])

    @property
    def s2(self):
        """ <S^2> value of the current wave function
            Uses the formula S^2 = S- S+ + Sz Sz + Sz, which corresponds to 
                <S^2> = <Sz> * (<Sz> + 1) + <Nb> - sum_pq G^{ab}_{pqqp} 
            where G^{ab}_{pqqp} is the alfa-beta component of the 2-RDM
        """
        return self.sz * (self.sz + 1)

    @property
    def gradient(self):
        """ Compute the gradient of the energy with respect to the orbital rotations"""
        return 2 * (self.gen_fock.T - self.gen_fock)[self.rot_idx]

    @property
    def hessian(self):
        ''' This method finds orb-orb part of the Hessian '''
        # Get generalised Fock and symmetrise
        F = self.gen_fock + self.gen_fock.T
        # Get one-electron matrix elements 
        h1e = self.mo_transform(self.integrals.oei_matrix(True))
        # Combine intermediates (Eq. 10.8.53 in Helgaker book) 
        Hess = 2 * self.get_Y_intermediate()
        for i in range(self.nmo):
            Hess[i,:,i,:] += 2 * self.mo_occ[i] * h1e
            Hess[:,i,:,i] -= F
        # Apply permutation symmetries
        Hess = Hess - Hess.transpose(1,0,2,3)
        Hess = Hess - Hess.transpose(0,1,3,2)
        # Reshape and return
        return (Hess[:, :, self.rot_idx])[self.rot_idx, :]
    
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

    @property 
    def dipole(self):
        """ Compute the dipole moment of the current wave function"""
        # Get the dipole integrals
        nucl_dip, ao_dip = self.integrals.dipole_matrix()
        # Return the combination
        return nucl_dip - np.einsum('xij,ji->x',ao_dip,self.dj)

        
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
            Ipqpq, Ipqqp = self.get_open_JK()
            matrix_print(self.exchange_matrix, title="Open-Shell Exchange Matrix <Ψ|Êpq Êqp|Ψ> - Np")
            matrix_print(Ipqqp[:self.nocc,self.ncore:self.nocc], title="Open-Shell Exchange Integrals <pq|qp>")
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
    
    def mo_transform(self, M):
        """ Transform a matrix from AO to MO basis """
        if(type(M) is tuple):
            return (self.mo_transform(Mi) for Mi in M)
        elif(type(M) is list):
            return [self.mo_transform(Mi) for Mi in M]
        elif(type(M) is dict):
            return {k:self.mo_transform(Mi) for k, Mi in M.items()}
        elif(type(M) is np.ndarray):
            if(len(M.shape)==3):
                return np.einsum('mp,imn,nq->ipq',self.mo_coeff,M,self.mo_coeff,optimize='optimal')
            elif(len(M.shape)==2):
                return np.einsum('mp,mn,nq->pq',self.mo_coeff,M,self.mo_coeff,optimize='optimal')
            else:
                raise ValueError("Matrix must be rank 2 or 3 for MO transformation")

    def hess_on_vec(self, vec):
        """ Compute the Hessian @ vec product directly, without forming the full Hessian matrix 
            This is more memory efficient and avoids any ERI computation.

            Inputs:
                vec : vector to be multiplied by the Hessian
            Returns:
                Hvec : result of Hessian @ vec product
        """
        # Antisymmetric step
        step = np.zeros((self.nmo,self.nmo))
        step[self.rot_idx] = vec
        step -= step.T

        ## 0. Initialize H @ vec
        Hvec = np.zeros_like(step)

        ## 1. One-electron part
        h1e = self.mo_transform(self.integrals.oei_matrix(True))
        Hvec += 2 * h1e @ np.einsum('sq,q->sq',step,self.mo_occ)

        ## 2. Generalised Fock part
        Ft = 0.5 * (self.gen_fock + self.gen_fock.T)
        Hvec -= step @ Ft + Ft @ step

        ## 3. Zeroth-order J/K part
        Jmo, Kmo = self.mo_transform((self.J,self.K))
        Hvec += 2 * np.einsum('p,ps->ps',self.mo_occ,step) @ Jmo
        # Core contribution
        for I in range(Kmo.shape[0]):
            Hvec[:self.ncore,:] -= 2 * step[:self.ncore,:] @ Kmo[I]
        for P, Pinds in enumerate(self.shell_indices):
            Hvec[Pinds,:] += step[Pinds,:] @ (2 * np.einsum('r,rqs->qs',self.beta[P],Kmo[1:]) - Kmo[0])
            
        ## 4. J/K part
        # Build first-order densities
        vd = np.zeros((self.nshell+1,self.nbsf,self.nbsf))
        vd[0] = np.linalg.multi_dot([self.mo_coeff,np.einsum('r,rs->rs',self.mo_occ,step),self.mo_coeff.T])
        # Symmetrise for efficiency
        vd[0] = 0.5 * (vd[0] + vd[0].T)
        for P in range(self.nshell):
            # Core contribution
            vd[P+1] -= np.linalg.multi_dot([self.mo_coeff[:,:self.ncore],step[:self.ncore,:],self.mo_coeff.T])
            # Active contribution
            for R, Rinds in enumerate(self.shell_indices):
                vd[P+1] += self.beta[P,R] * np.linalg.multi_dot([self.mo_coeff[:,Rinds],step[Rinds,:],self.mo_coeff.T])
            vd[P+1] = 0.5 * (vd[P+1] + vd[P+1].T)
        # Build J and K matrices from first-order densities
        J1, _, K1 = self.mo_transform(self.integrals.build_JK(vd,vd,hermi=1,Kxc=True))
        # Contribution to Hvec
        Hvec += 4 * np.einsum('p,qp->pq',self.mo_occ,J1[0]) 
        # Don't include transpose as density was symmetrised
        Hvec[:self.ncore,:] -= 4 * K1[0][:self.ncore,:]
        for P, Pinds in enumerate(self.shell_indices):
            Hvec[Pinds,:] += 4 * K1[P+1][Pinds,:]
        
        ## 5. Antisymmetrise and return
        Hvec = Hvec - Hvec.T
        return Hvec[self.rot_idx]

    def get_rdm12(self,only_occ=True):
        """ Compute the 1- and 2-electron reduced matrices from the shell coupling in occupied space
            returns: 
                dm1: 1-electron reduced density matrix
                dm2: 2-electron reduced density matrix
        """
        # Numbers 
        nocc = self.nocc
        ncore = self.ncore
        nmo = self.nmo

        # 1-RDM
        if only_occ:
            dm1 = np.diag(self.mo_occ[:nocc])
        else:
            dm1 = np.diag(self.mo_occ)

        # 2-RDM
        if only_occ:
            dm2 = np.zeros((nocc,nocc,nocc,nocc))
        else:
            dm2 = np.zeros((nmo,nmo,nmo,nmo))

        for p in range(ncore):
            for q in range(ncore):
                if(p==q):
                    dm2[p,p,p,p] = 2
                else:
                    dm2[p,q,p,q] = 4
                    dm2[p,q,q,p] = - 2
            for w in range(ncore,nocc):
                dm2[p,w,p,w] = 2
                dm2[p,w,w,p] = -1
                dm2[w,p,w,p] = 2
                dm2[w,p,p,w] = -1

        for W, sW in enumerate(self.shell_indices):
            for V, sV in enumerate(self.shell_indices):
                for w in sW:
                    for v in sV:
                        if(w==v):
                            dm2[w,w,w,w] = 0
                        else:
                            dm2[w,v,w,v] = 1 
                            dm2[w,v,v,w] = self.beta[W,V]
        return dm1, dm2

    def update(self):
        """ Update the integrals with current set of orbital coefficients"""
        # Update density matrices (AO basis)
        self.vd = self.get_density_matrices()
        #self.dj, self.dk, self.vd = self.get_density_matrices()
        # Update JK matrices (AO basis) 
        self.J, self.K = self.get_JK_matrices(self.vd)
        # Get Fock matrix (AO basis)
        self.fock_vir  = (  self.integrals.oei_matrix(True) + self.J - self.K[0] 
                          + np.einsum('i,ipq->pq',self.anion_dbeta,self.K[1:]) )
        # Get generalized Fock matrices
        self.gen_fock, self.fock_shell = self.get_generalised_fock()
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
        if hasattr(self, 'hess_index'):
            hindices = self.hess_index
        else:
            hindices = (0,0)
        with open(tag+".solution", "w") as F:
            F.write(f"{self.energy:18.12f} {hindices[0]:5d} {hindices[1]:5d} {self.s2:12.6f} {self.spin_coupling:s}\n")

        # Save the civector dump
        self.write_cidump(tag)
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


    def read_from_orca(self,json_file):
        """ Read a set of CSF coefficients from ORCA gbw file.
            This requires the orca_2json executable to be available and spin_coupling 
            must be set in the Quantel input file.
        """
        import json
        # Read ORCA Json file
        with open(json_file, 'r') as f:
            data = json.load(f)
        mo_read = np.array([value['MOCoefficients'] for value in data['Molecule']['MolecularOrbitals']['MOs']]).T
        
        # TODO: For now, we have a temporary fix to change the sign of the f+3 and f-3 orbitals, 
        #       which appear to be inconsistent between Libint and ORCA
        orb_labels = data['Molecule']['MolecularOrbitals']['OrbitalLabels']
        phase_shift = []
        for i, l in enumerate(orb_labels):
            if (r'f+3' in l) or (r'f-3' in l):
                phase_shift.append(i)
        mo_read[phase_shift,:] *= -1
         
        # Initialise the wave function
        self.initialise(mo_read, spin_coupling=self.spin_coupling)   

        # Check the input
        if mo_read.shape[0] != self.nbsf:
            raise ValueError("Inccorect number of AO basis functions in file")
        if mo_read.shape[1] < self.nocc:
            raise ValueError("Insufficient orbitals in file to represent occupied orbitals")
        if mo_read.shape[1] > self.nmo:
            raise ValueError("Too many orbitals in file")

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
        if(reorder and (self.spin_coupling != '')):
            Cguess[:,self.ncore:self.nocc] = csf_reorder_orbitals(self.integrals,self.exchange_matrix,
                                                                  np.copy(Cguess[:,self.ncore:self.nocc]))

        # Initialise the CSF object with the guess coefficients.
        self.initialise(Cguess, spin_coupling=self.spin_coupling)
        return


    def restore_last_step(self):
        """ Restore MO coefficients to previous step"""
        self.mo_coeff = self.mo_coeff_save.copy()
        self.update()


    def save_last_step(self):
        """ Save MO coefficients"""
        self.mo_coeff_save = self.mo_coeff.copy()


    def take_step(self, step):
        """ Take a step in the orbital space"""
        self.save_last_step()
        self.rotate_orb(step[:self.nrot])
        self.update()


    def rotate_orb(self, step):
        """ Rotate molecular orbital coefficients with a step"""
        orb_step = np.zeros((self.nmo, self.nmo))
        orb_step[self.rot_idx] = step
        self.mo_coeff = np.dot(self.mo_coeff, scipy.linalg.expm(orb_step - orb_step.T))


    def transform_vector(self,vec,step,X=None):
        """ Perform orbital rotation for vector in tangent space"""
        # Construct transformation matrix
        orb_step = np.zeros((self.nmo, self.nmo))
        orb_step[self.rot_idx] = step
        orb_step = orb_step - orb_step.T

        # Build vector in antisymmetric form
        kappa = np.zeros((self.nmo, self.nmo))
        kappa[self.rot_idx] = vec
        kappa = kappa - kappa.T

        # Apply transformation
        kappa = flag_transport(kappa,orb_step,self.invariant)

        # 05/10/2024 - This is the old formula, only applicable for a small step
        #Q = scipy.linalg.expm(0.5 * (orb_step - orb_step.T))
        #kappa = kappa @ Q
        #kappa = Q.T @ kappa

        # Also apply horizontal transform
        if not X is None:
            kappa = kappa @ X
            kappa = X.T @ kappa

        # Return transformed vector
        return kappa[self.rot_idx]
    

    def get_density_matrices(self):
        """ Compute total density matrix and relevant matrices for K build"""
        # Initialise memory
        vd = np.zeros((1+self.nshell,self.nbsf,self.nbsf))
        # Core contribution       
        vd[0] = 2 * self.mo_coeff[:,:self.ncore] @ self.mo_coeff[:,:self.ncore].T
        # Contribution from each open shell
        for W, Wshell in enumerate(self.shell_indices):
            vd[W+1] = self.mo_coeff[:,Wshell] @ self.mo_coeff[:,Wshell].T
        return vd

        ## Initialise densities
        #vd = np.zeros((1+self.nopen,self.nbsf,self.nbsf))
        ## Core contribution       
        #vd[0] = 2 * self.mo_coeff[:,:self.ncore] @ self.mo_coeff[:,:self.ncore].T
        ## Contribution from each active orbital
        #for id, i in enumerate(range(self.ncore,self.nocc)):
        #    vd[id+1] = self.mo_occ[i] * np.outer(self.mo_coeff[:,i],self.mo_coeff[:,i])

        ## Extract shell densities
        #dj  = np.einsum('kpq->pq',vd)
        #dk = np.zeros((self.nshell+1,self.nbsf,self.nbsf))
        #dk[0] = vd[0].copy()
        #for Ishell in range(self.nshell):
        #    shell = [1+i-self.ncore for i in self.shell_indices[Ishell]]
        #    dk[Ishell+1] += np.einsum('vpq->pq',vd[shell])
        #return dj, dk, vd

    def get_spin_density(self):
        """ Compute the alfa and beta density matrices"""
        dm_tmp = np.einsum('kpq->pq',self.vd[1:])
        rho_a, rho_b = 0.5 * self.vd[0], 0.5 * self.vd[0]
        if(self.nopen > 0):
            rho_a += (0.5 + self.sz / self.nopen) * dm_tmp
            rho_b += (0.5 - self.sz / self.nopen) * dm_tmp
        return rho_a, rho_b


    def get_JK_matrices(self, vd):
        ''' Compute the JK matrices and diagonal two-electron integrals

            This function also computes the density matrices (maybe redundant)

            Input:
                vd: Density matrices for core and each open orbital

            Returns:
                J: Total Coulomb matrix
                K: Exchange matrices for each shell
        '''
        # Build the integrals with incremental JK build
        #if hasattr(self, "vd_last"):
        #    # Compute difference density
        #    _vd = vd - self.vd_last
        #    # Compute difference J, K, and Ipqqp
        #    _vJ, _, _vK = self.integrals.build_JK(_vd,_vd,hermi=1,Kxc=True)
        #    # Compute incremental update to J and K
        #    vJ = self.vJ_last + _vJ
        #    vK = self.vK_last + _vK
        #else:
        vJ, _, vK = self.integrals.build_JK(vd,vd,hermi=1,Kxc=True)

        # Save last elements
        self.vd_last = vd.copy()
        self.vJ_last = vJ.copy()
        self.vK_last = vK.copy()

        # Get the total J matrix
        J = np.einsum('kpq->pq',vJ)

        return J, vK


    def get_generalised_fock(self):
        """ Compute the generalised Fock matrix in MO basis"""
        # Build fock matrix for each shell 
        Fshell = np.zeros((1+self.nshell,self.nmo,self.nmo))
        # Also build total generalised Fock matrix
        F = np.zeros((self.nmo, self.nmo)) 

        # Core contribution
        Fcore_ao = 2 * (self.integrals.oei_matrix(True) + self.J 
                      - 0.5 * np.sum(self.K[i] for i in range(self.nshell+1)))
        # AO-to-MO transformation
        Fshell[0] = np.linalg.multi_dot([self.mo_coeff.T, Fcore_ao, self.mo_coeff])
        F[:self.ncore,:] = Fshell[0][:self.ncore,:]

        # Open-shell contributions
        for W, shell in enumerate(self.shell_indices):
            # One-electron matrix, Coulomb and core exchange
            Fw_ao = self.integrals.oei_matrix(True) + self.J - 0.5 * self.K[0]
            # Different shell exchange
            Fw_ao += np.einsum('v,vpq->pq',self.beta[W],self.K[1:])
            # AO-to-MO transformation
            Fshell[W+1] = np.linalg.multi_dot([self.mo_coeff.T, Fw_ao, self.mo_coeff])
            F[shell,:] = Fshell[W+1][shell,:]
        
        return F, Fshell


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
        # Y_imjn
        Y[:ncore,:,:ncore,:] += 8 * np.einsum('mnij->imjn',ppoo[:,:,:ncore,:ncore]) 
        Y[:ncore,:,:ncore,:] -= 2 * self.integrals.hybrid_K * np.einsum('mnji->imjn',ppoo[:,:,:ncore,:ncore])
        Y[:ncore,:,:ncore,:] -= 2 * self.integrals.hybrid_K * np.einsum('mjni->imjn',popo[:,:ncore,:,:ncore])
        for i in range(ncore):
            Y[i,:,i,:] += 2 * Jmn - Kmn

        # Y_imwn
        Y[:ncore,:,ncore:nocc,:] = (4 * ppoo[:,:,:ncore,ncore:nocc].transpose(2,0,3,1)
                                      - self.integrals.hybrid_K * ppoo[:,:,ncore:nocc,:ncore].transpose(3,0,2,1)
                                      - self.integrals.hybrid_K * popo[:,ncore:nocc,:,:ncore].transpose(3,0,1,2))
        Y[ncore:nocc,:,:ncore,:] = Y[:ncore,:,ncore:nocc,:].transpose(2,3,0,1)

        # Y_wmvn
        for W in range(self.nshell):
            wKmn = np.einsum('v,vmn->mn',self.beta[W], vKmn[1:])
            for V in range(W,self.nshell):
                for w in self.shell_indices[W]:
                    for v in self.shell_indices[V]:
                        Y[w,:,v,:] = 2 * ppoo[:,:,w,v] + self.integrals.hybrid_K * self.beta[W,V] * (ppoo[:,:,v,w] + popo[:,v,:,w])
                        if(w==v):
                            Y[w,:,w,:] = Y[w,:,w,:] + Jmn - 0.5 * vKmn[0] + wKmn
                        else:
                            Y[v,:,w,:] = Y[w,:,v,:].T
        return Y


    def get_open_JK(self):
        """ Compute <pq|pq> and <pq|qp> for open-shell orbitals p"""
        vd = np.einsum('mp,np->pmn', self.mo_coeff[:,self.ncore:self.nocc], self.mo_coeff[:,self.ncore:self.nocc])
        vJ, vK = self.integrals.build_JK(vd,vd,hermi=1,Kxc=False)
        Ipqpq = np.einsum('pmn,mq,nq->pq', vJ, self.mo_coeff, self.mo_coeff, optimize='optimal')
        Ipqqp = np.einsum('pmn,mq,nq->pq', vK, self.mo_coeff, self.mo_coeff, optimize='optimal')
        return Ipqpq, Ipqqp 

    
    def get_preconditioner(self,abs=True):
        """Compute approximate diagonal of Hessian"""
        # Initialise approximate preconditioner
        Q = np.zeros((self.nmo,self.nmo))

        # Core contribution
        Fcore_diag = self.fock_shell[0].diagonal()
        Q[:self.ncore,:] += 2 * (Fcore_diag[None,:] - Fcore_diag[:self.ncore,None]) 

        # Active contribution
        for W, shell in enumerate(self.shell_indices):
            Fshell_diag = self.fock_shell[W+1].diagonal()
            Q[shell,:] += 2 * (Fshell_diag[None,:] - Fshell_diag[shell,None])
        
        # Antisymmetrise
        Q = Q + Q.T

        if(self.advanced_preconditioner):
            # HGAB 08-01-2025: We remove these contributions to avoid the Nshell * N^4 cost of computing
            # active-active Coulomb and exchange integrals. This means we no longer compute Ipqqp and Ipqpq.
            Ipqpq, Ipqqp = self.get_open_JK()

            # Compute two-electron corrections involving active orbitals
            Acoeff = Ipqqp
            for q in range(self.ncore,self.nocc):
                for p in range(q):
                    Q[q,p] += 4 * (self.mo_occ[p]-self.mo_occ[q])**2 * Acoeff[q-self.ncore,p]
                for p in range(q+1,self.nmo):
                    Q[p,q] += 4 * (self.mo_occ[p]-self.mo_occ[q])**2 * Acoeff[q-self.ncore,p]

            Bcoeff = self.integrals.hybrid_K * (Ipqpq + Ipqqp)
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

        return np.abs(Q[self.rot_idx]) if abs else Q[self.rot_idx]

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

    def invariant_indices(self):
        """ This function creates a matrix of boolean of size (norb,norb).
            A True element means that this rotation should be taken into
            account during the optimization. Taken from pySCF.mcscf.casscf
        """
        mask = np.zeros((self.nmo, self.nmo), dtype=bool)
        mask[:self.ncore,:self.ncore] = True
        for W in self.shell_indices:
            mask[W,W] = True
        mask[self.nocc:,self.nocc:] = True
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
            np.ones((self.nopen, self.nopen), dtype=bool), k=-1)
        
        # Modify for genealogical coupling
        if self.spin_coupling != None:
            mask[self.ncore:self.nocc, self.ncore:self.nocc] = self.edit_mask_by_gcoupling(
                mask[self.ncore:self.nocc,self.ncore:self.nocc])
            
        # Account for any frozen orbitals   
        if frozen != None:
            if isinstance(frozen, (int, np.integer)):
                mask[:frozen] = mask[:, :frozen] = False
            else:
                frozen = np.asarray(frozen)
                mask[frozen] = mask[:, frozen] = False
        return mask


    def localise_orbitals(self,pop_method='becke'):
        """ Localise occupied orbitals for each shell using Pipek-Mezey localisation"""
        # Localise each shell individually
        for shell in self.shell_indices:
            self.mo_coeff[:,shell] = localise_orbitals(self.integrals.molecule(),
                                                       self.mo_coeff[:,shell])
        self.update()


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

    def get_active_integrals(self):
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
        foo = 0.5 * self.gen_fock[:self.ncore,:self.ncore]
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
        Cvir = self.mo_coeff[:,self.nocc:].copy()
        fvv = np.linalg.multi_dot([Cvir.T, self.fock_vir, Cvir])
        self.mo_energy[self.nocc:], Qvv = stable_eigh(fvv)
        Q[self.nocc:,self.nocc:] = Qvv

        # Apply transformation
        if(np.linalg.det(Q) < 0): Q[:,0] *= -1
        self.mo_coeff = self.mo_coeff @ Q
        
        # Update generalised Fock matrix and diagonal approximations
        self.update()
        return Q

    def get_anion_exchange(self):
        """Compute the exchange coupling coefficients for adding + or - electron on average."""
        # Addition of '+' electron
        _,new_indices = get_shells(0,self.spin_coupling+'+')
        dbeta = get_shell_exchange(0,new_indices,self.spin_coupling+'+')[-1,:self.nshell]

        # Addition of '-' electron if possible
        if(self.sz > 0):
            _,new_indices = get_shells(0,self.spin_coupling+'-')
            dbeta += get_shell_exchange(0,new_indices,self.spin_coupling+'-')[-1,:self.nshell]
            # Divide by 2 to give average of adding + or - electron
            dbeta *= 0.5

        return dbeta
