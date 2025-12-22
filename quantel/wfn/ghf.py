#!/usr/bin/python3
# Author: Hugh G. A. Burton

import numpy as np
import scipy.linalg
import h5py
from quantel.utils.linalg import orthogonalise, matrix_print
from .wavefunction import Wavefunction
import quantel
from pyscf.tools import cubegen

class GHF(Wavefunction):
    """ Generalised Hartree-Fock method

        Inherits from the Wavefunction abstract base class with pure virtual properties:
            - energy
            - gradient
            - hessian 
            - take_step
            - save_last_step
            - restore_step
    """
    def __init__(self, integrals, verbose=0, mom_method=None):
        """Initialise Restricted Hartree-Fock wave function
               integrals : quantel integral interface
               verbose   : verbosity level
        """
        self.integrals = integrals
        self.nelec     = integrals.molecule().nalfa() + integrals.molecule().nbeta()
        if(self.integrals.xc is not None):
            raise NotImplementedError("GHF not compatible with exchange-correlation functional")

        # Get number of basis functions and linearly independent orbitals
        self.nbsf      = integrals.nbsf()
        self.nmo       = 2*integrals.nmo()
        self.with_xc    = (type(integrals) is not quantel.lib._quantel.LibintInterface)
        if(self.with_xc): self.with_xc = (integrals.xc is not None)

        # For now, we assume that the number of alpha and beta electrons are the same
        self.nocc      = self.nelec
        self.verbose   = verbose
        self.mom_method = mom_method

        # Setup the indices for relevant orbital rotations
        self.rot_idx   = self.uniq_var_indices() # Indices for orbital rotations
        self.nrot      = np.sum(self.rot_idx) # Number of orbital rotations

        # Define the orbital energies and coefficients
        self.mo_coeff   = None
        self.mo_energy = None

        # Define extended overlap matrix
        self.ghf_overlap = np.kron(np.eye(2),integrals.overlap_matrix())
        self.ghf_X = np.kron(np.eye(2), integrals.orthogonalization_matrix())
    
    def initialise(self, mo_guess, ci_guess=None):
        """Initialise the wave function with a set of molecular orbital coefficients"""
        # Make sure orbitals are orthogonal
        self.mo_coeff = orthogonalise(mo_guess, self.ghf_overlap)
        # Set initial orbital occupation
        if(self.mom_method == 'IMOM'):
            self.Cinit = self.mo_coeff.copy()
        # Update the density and Fock matrices
        self.update()

    @property
    def dim(self):
        """Get the number of degrees of freedom"""
        return self.nrot

    @property
    def energy(self):
        """Get the energy of the current RHF state"""
        # Get density spin component
        Da = self.dens[:self.nbsf,:self.nbsf]
        Db = self.dens[self.nbsf:,self.nbsf:]
        # Nuclear potential
        En  = self.integrals.scalar_potential()
        # One-electron energy
        E1  = np.einsum('pq,pq', self.integrals.oei_matrix(True), Da, optimize="optimal")
        E1 += np.einsum('pq,pq', self.integrals.oei_matrix(False), Db, optimize="optimal")
        # Two-electron energy
        E2 = 0.5 * np.einsum('pq,pq', self.JK, self.dens, optimize="optimal")
        # Save components
        self.energy_components = dict(Nuclear=En, One_Electron=E1, Two_Electron=E2, Exchange_Correlation=0)
        return En + E1 + E2

    @property 
    def sz(self):
        """Get the expectation value of the spin operator S^z"""
        S = self.integrals.overlap_matrix()
        # Get density spin component
        Da = self.dens[:self.nbsf,:self.nbsf]
        Db = self.dens[self.nbsf:,self.nbsf:]
        # Compute <Sz>
        return 0.5 * (np.trace(S @ Da) - np.trace(S @ Db))
    
    @property
    def s2(self):
        """Get the spin of the current RHF state"""
        raise NotImplementedError("Spin S^2 not implemented for GHF")
        return 0 # All RHF states have spin 0

    @property
    def gradient(self):
        """Get the energy gradient with respect to the orbital rotations"""
        g = 2 * np.linalg.multi_dot([self.mo_coeff.T, self.fock, self.mo_coeff])
        return g[self.rot_idx]

    @property
    def hessian(self):
        """Compute the internal RHF orbital Hessian"""
        # Number of occupied and virtual orbitals
        no = self.nocc

        # Compute Fock matrix in MO basis 
        Fmo = np.linalg.multi_dot([self.mo_coeff.T, self.fock, self.mo_coeff])

        # Get occupied and virtual orbital coefficients
        Cao = self.mo_coeff[:self.nbsf,:no].copy()
        Cbo = self.mo_coeff[self.nbsf:,:no].copy()
        Cav = self.mo_coeff[:self.nbsf,no:].copy()
        Cbv = self.mo_coeff[self.nbsf:,no:].copy()

        # Compute ao_to_mo integral transform
        eri_abij = (  self.integrals.tei_ao_to_mo(Cav,Cav,Cao,Cao,True,False)
                    + self.integrals.tei_ao_to_mo(Cav,Cbv,Cao,Cbo,True,False)
                    + self.integrals.tei_ao_to_mo(Cbv,Cav,Cbo,Cao,True,False)
                    + self.integrals.tei_ao_to_mo(Cbv,Cbv,Cbo,Cbo,True,False))
        eri_aibj = (  self.integrals.tei_ao_to_mo(Cav,Cao,Cav,Cao,True,False)
                    + self.integrals.tei_ao_to_mo(Cav,Cbo,Cav,Cbo,True,False)
                    + self.integrals.tei_ao_to_mo(Cbv,Cao,Cbv,Cao,True,False)
                    + self.integrals.tei_ao_to_mo(Cbv,Cbo,Cbv,Cbo,True,False))

        # Initialise Hessian matrix
        hessian = np.zeros((self.nmo,self.nmo,self.nmo,self.nmo))

        # Compute Fock contributions
        for i in range(no):
            hessian[no:,i,no:,i] += 2 * Fmo[no:,no:]
        for a in range(no,self.nmo):
            hessian[a,:no,a,:no] -= 2 * Fmo[:no,:no]

        # Compute two-electron contributions
        hessian[no:,:no,no:,:no] += 4 * np.einsum('abij->aibj', eri_abij, optimize="optimal")
        hessian[no:,:no,no:,:no] -= 2 * np.einsum('aibj->aibj', eri_aibj, optimize="optimal")
        hessian[no:,:no,no:,:no] -= 2 * np.einsum('abji->aibj', eri_abij, optimize="optimal")

        # Return suitably shaped array
        return (hessian[:,:,self.rot_idx])[self.rot_idx,:]


    def hess_on_vec(self,X):
        """ Compute the action of Hessian on a vector X"""
        # Reshape X into matrix form
        Xai = np.reshape(X, (self.nmo-self.nocc,self.nocc))
        # Access occupied and virtual orbitals
        Ci = self.mo_coeff[:,:self.nocc]
        Ca = self.mo_coeff[:,self.nocc:]

        # First order density change
        Dia = np.einsum('pa,ai,qi->pq', Ca, Xai, Ci, optimize="optimal")
        Daa = Dia[:self.nbsf,:self.nbsf]
        Dab = Dia[:self.nbsf,self.nbsf:]
        Dba = Dia[self.nbsf:,:self.nbsf]
        Dbb = Dia[self.nbsf:,self.nbsf:]
        # First order JK integrals
        (Jaa,_,_,Jbb), (Kaa,Kab,Kba,Kbb) = self.integrals.build_JK([Daa,Dab,Dba,Dbb],[Daa,Dab,Dba,Dbb],hermi=0,Kxc=False)
        Jia = np.zeros((2*self.nbsf, 2*self.nbsf))
        Jia[:self.nbsf,:self.nbsf] = Jaa + Jbb
        Jia[self.nbsf:,self.nbsf:] = Jaa + Jbb
        Kia = np.block([[Kaa,Kab],
                        [Kba,Kbb]])
        
        # Fock contributions
        Fba = Ca.T @ self.fock @ Ca
        Fij = Ci.T @ self.fock @ Ci
        HX = 2 * (Fba @ Xai - Xai @ Fij)
        kernel = 4 * Jia  - 2 * (Kia + Kia.T)
        HX += np.einsum('pa,qp,qi->ai', Ca, kernel, Ci, optimize="optimal")
        return HX.ravel()


    def print(self,verbose=1):
        """ Print details about the state energy and orbital coefficients

            Inputs:
                verbose : level of verbosity
                          0 = No output
                          1 = Print energy components and spin
                          2 = Print energy components, spin, and occupied orbital coefficients
                          3 = Print energy components, spin, and all orbital coefficients
                          4 = Print energy components, spin, Fock matrix, and all orbital coefficients 
        """
        if(verbose > 0):
            print("\n ---------------------------------------------")
            print(f"         Total Energy = {self.energy:14.8f} Eh")
            for key, value in self.energy_components.items():
                print(f" {key.replace('_',' '):>20s} = {value:14.8f} Eh")
            print(" ---------------------------------------------")
            print(f"        <Sz> = {self.sz:5.2f}")
            #print(f"        <S2> = {self.s2:5.2f}")
        if(verbose > 2):
            matrix_print(self.mo_coeff[:,:self.nocc], title="Occupied Orbital Coefficients")
        if(verbose > 3):
            matrix_print(self.mo_coeff[:,self.nocc:], title="Virtual Orbital Coefficients", offset=self.nocc)
        if(verbose > 4):
            matrix_print(self.fock, title="Fock Matrix (AO basis)")
        print()

    def save_to_disk(self,tag):
        """Save object to disk with prefix 'tag'"""
        # Canonicalise orbitals
        self.canonicalize()
 
         # Save hdf5 file with MO coefficients, orbital energies, energy, and spin
        with h5py.File(tag+".hdf5", "w") as F:
            F.create_dataset("mo_coeff", data=self.mo_coeff)
            F.create_dataset("mo_energy", data=self.mo_energy)
            F.create_dataset("energy", data=self.energy)
            F.create_dataset("s2", data=self.s2)    
        
        # Save numpy txt file with energy and Hessian indices
        hindices = self.get_hessian_index()
        with open(tag+".solution", "w") as F:
            F.write(f"{self.energy:18.12f} {hindices[0]:5d} {hindices[1]:5d} {self.s2:12.6f}\n")

    def read_from_disk(self,tag):
        """Read object from disk with prefix 'tag'"""
        # Read MO coefficients from hdf5 file
        with h5py.File(tag+".hdf5", "r") as F:
            mo_read = F["mo_coeff"][:]
        # Initialise object
        self.initialise(mo_read)

    def copy(self):
        """Return a copy of the current RHF object"""
        them = GHF(self.integrals, verbose=self.verbose)
        them.initialise(self.mo_coeff)
        return them

    def overlap(self, them):
        """Compute the (nonorthogonal) many-body overlap with another RHF wavefunction (them)"""
        if(self.nocc != them.nocc):
            return 0
        nocc = self.nocc
        S = np.linalg.multi_dot([self.mo_coeff[:,:nocc].T, self.ghf_overlap, them.mo_coeff[:,:nocc]])
        return np.linalg.det(S)**2

    def hamiltonian(self, them):
        """Compute the (nonorthogonal) many-body Hamiltonian coupling with another RHF wavefunction (them)"""
        raise NotImplementedError("RHF Hamiltonian not implemented")

    def update(self):
        """Update the 1RDM and Fock matrix for the current state"""
        self.get_density()
        self.get_fock()


    def get_density(self):
        """Compute the 1RDM for the current state in AO basis"""
        Cocc = self.mo_coeff[:,:self.nocc]
        self.dens = np.dot(Cocc, Cocc.T)
        # List of density spin components
        self.vd = np.zeros((3,self.nbsf,self.nbsf))
        self.vd[0] = self.dens[:self.nbsf,:self.nbsf]  # Alpha-alpha
        self.vd[1] = self.dens[:self.nbsf,self.nbsf:]  # Alpha-beta
        self.vd[2] = self.dens[self.nbsf:,self.nbsf:]  # Beta-beta


    def get_fock(self):
        """Compute the Fock matrix for the current state"""
        # Get JK integrals
        self.vJ, self.vK = self.integrals.build_JK(self.vd,self.vd,Kxc=False)
        # Construct the Coulomb matrix
        self.J = np.zeros((2*self.nbsf, 2*self.nbsf))
        self.J[:self.nbsf,:self.nbsf] = self.vJ[0] + self.vJ[2]
        self.J[self.nbsf:,self.nbsf:] = self.vJ[0] + self.vJ[2]
        # Construct the exchange matrix
        self.K = np.zeros((2*self.nbsf, 2*self.nbsf))
        self.K[:self.nbsf,:self.nbsf] = self.vK[0] # aa
        self.K[self.nbsf:,self.nbsf:] = self.vK[2] # bb
        self.K[:self.nbsf,self.nbsf:] = self.vK[1] # ab
        self.K[self.nbsf:,:self.nbsf] = self.vK[1].T # ba
        # Compute the Coulomb and Exchange matrices
        self.fock = np.kron(np.eye(2), self.integrals.oei_matrix(True)) + self.J - self.K
        self.JK = self.J - self.K
        # Vectorised format of the Fock matrix 
        return self.fock.T.reshape((-1))


    def canonicalize(self):
        """Diagonalise the occupied and virtual blocks of the Fock matrix"""
        # Initialise orbital energies
        self.mo_energy = np.zeros(self.nmo)
        # Get Fock matrix in MO basis
        Fmo = np.linalg.multi_dot([self.mo_coeff.T, self.fock, self.mo_coeff])
        # Extract occupied and virtual blocks
        Focc = Fmo[:self.nocc,:self.nocc]
        Fvir = Fmo[self.nocc:,self.nocc:]
        # Diagonalise the occupied and virtual blocks
        self.mo_energy[:self.nocc], Qocc = np.linalg.eigh(Focc)
        self.mo_energy[self.nocc:], Qvir = np.linalg.eigh(Fvir)
        # Build the canonical MO coefficients
        self.mo_coeff[:,:self.nocc] = np.dot(self.mo_coeff[:,:self.nocc], Qocc)
        self.mo_coeff[:,self.nocc:] = np.dot(self.mo_coeff[:,self.nocc:], Qvir)
        self.update()
        # Get orbital occupation
        self.mo_occ = np.zeros(self.nmo)
        self.mo_occ[:self.nocc] = 2.0
        # Combine full transformation matrix
        Q = np.zeros((self.nmo,self.nmo))
        Q[:self.nocc,:self.nocc] = Qocc
        Q[self.nocc:,self.nocc:] = Qvir
        return Q


    def get_preconditioner(self):
        """Compute approximate diagonal of Hessian"""
        # Get Fock matrix in MO basis
        fock_mo = np.linalg.multi_dot([self.mo_coeff.T, self.fock, self.mo_coeff])
        # Initialise approximate preconditioner
        Q = np.zeros((self.nmo,self.nmo))
        # Include dominate generalised Fock matrix terms
        for p in range(self.nmo):
            for q in range(p):
                Q[p,q] = 2 * (fock_mo[p,p] - fock_mo[q,q])
        return np.abs(Q[self.rot_idx])


    def diagonalise_fock(self):
        """Diagonalise the Fock matrix"""
        # Get the orthogonalisation matrix
        # Project to linearly independent orbitals
        Ft = np.linalg.multi_dot([self.ghf_X.T, self.fock, self.ghf_X])
        # Diagonalise the Fock matrix
        Et, Ct = np.linalg.eigh(Ft)
        # Transform back to the original basis
        Cnew = np.dot(self.ghf_X, Ct)
        
        # Select occupied orbitals using MOM if specified
        if(self.mom_method =='MOM'):
            Cold = self.mo_coeff.copy()
            self.mo_coeff = self.mom_select(Cold,Cnew)
        elif(self.mom_method == 'IMOM'):
            self.mo_coeff = self.mom_select(self.Cinit,Cnew)
        else:
            self.mo_coeff = Cnew

        # Save current orbital energies
        self.mo_energy = self.mo_coeff.T @ self.fock @ self.mo_coeff
        # Update density and Fock matrices
        self.update()


    def transform_vector(self,vec,step,X=None):
        """ Perform orbital rotation for vector in tangent space"""
        # Build vector in antisymmetric form
        kappa = np.zeros((self.nmo, self.nmo))
        kappa[self.rot_idx] = vec
        kappa = kappa - kappa.T
        # Only horizontal transformations leave unchanged
        if not X is None:
            kappa = kappa @ X
            kappa = X.T @ kappa
        return kappa[self.rot_idx]

    def try_fock(self, fock_vec):
        """Try an extrapolated Fock matrix and update the orbital coefficients"""
        self.fock = fock_vec.reshape((2*self.nbsf,2*self.nbsf))
        self.diagonalise_fock()

    def get_diis_error(self):
        """Compute the DIIS error vector and DIIS error"""
        err_vec  = np.linalg.multi_dot([self.fock, self.dens, self.ghf_overlap])
        err_vec -= err_vec.T
        return err_vec.ravel(), np.linalg.norm(err_vec)

    def restore_last_step(self):
        """Restore orbital coefficients to the previous step"""
        self.mo_coeff = self.mo_coeff_save.copy()
        self.update()

    def save_last_step(self):
        """Save the current orbital coefficients"""
        self.mo_coeff_save = self.mo_coeff.copy()

    def take_step(self,step):
        """Take a step in the orbital space"""
        self.save_last_step()
        self.rotate_orb(step[:self.nrot])

    def rotate_orb(self,step): 
        """Rotate molecular orbital coefficients with a step"""
        # Build the anti-symmetric step matrix
        K = np.zeros((self.nmo,self.nmo))
        K[self.rot_idx] = step
        # Build the unitary transformation
        Q = scipy.linalg.expm(K - K.T)
        # Transform the coefficients
        self.mo_coeff = np.dot(self.mo_coeff, Q)
        # Update the density and fock matrices
        self.update()

    def uniq_var_indices(self):
        """Create a matrix of boolean of size (nbsf,nbsf). 
           A True element means that this rotation should be taken into account during the optimization.
        """
        # Include only occupied-virtual rotations
        mask = np.zeros((self.nmo,self.nmo), dtype=bool)
        mask[self.nocc:,:self.nocc] = True
        return mask
    
    def get_orbital_guess(self, method="gwh"):
        """Get a guess for the molecular orbital coefficients"""
        # Get one-electron integrals and overlap matrix 
        h1e = self.integrals.oei_matrix(True)
        s = self.integrals.overlap_matrix()
        
        # Build guess Fock matrix
        if(method.lower() == "core"):
            # Use core Hamiltonian as guess
            self.fock = np.kron(np.eye(2), h1e.copy())
        elif(method.lower() == "gwh"):
            # Build GWH guess Hamiltonian
            K = 1.75
            
            self.fock = np.zeros((self.nbsf,self.nbsf))
            for i in range(self.nbsf):
                for j in range(self.nbsf):
                    self.fock[i,j] = 0.5 * (h1e[i,i] + h1e[j,j]) * s[i,j]
                    if(i!=j):
                        self.fock[i,j] *= 1.75
            self.fock = np.kron(np.eye(2), self.fock)
            
        else:
            raise NotImplementedError(f"Orbital guess method {method} not implemented")
        
        # Project to linearly independent orbitals
        Ft = np.linalg.multi_dot([self.ghf_X.T, self.fock, self.ghf_X])
        # Diagonalise the Fock matrix
        Et, Ct = np.linalg.eigh(Ft)
        # Transform back to the original basis
        Cinit = np.dot(self.ghf_X, Ct)
        # Get orbital coefficients by diagonalising Fock matrix
        self.initialise(Cinit)

    def deallocate(self):
        pass
        
    def approx_hess_on_vec(self, vec, eps=1e-3):
        """ Compute the approximate Hess * vec product using forward finite difference """
        # Get current gradient
        g0 = self.gradient.copy()
        # Save current position
        mo_save = self.mo_coeff.copy()
        # Get forward gradient
        self.take_step(eps * vec)
        g1 = self.gradient.copy()
        # Restore to origin
        self.mo_coeff = mo_save.copy()
        self.update()
        # Parallel transport back to current position
        g1 = self.transform_vector(g1, - eps * vec)
        # Get approximation to H @ sk
        return (g1 - g0) / eps

    def compare_density(self, them, complex_rot=False):
        """Compute distance metric assumming same if related by spin rotation"""
        # Get density matrices for both states
        self.get_density()
        them.get_density()
        Px = self.ghf_overlap @ self.dens
        Pw = self.ghf_overlap @ them.dens

        # Compute spin component
        Xaa = Px[:self.nbsf,:self.nbsf]
        Xab = Px[:self.nbsf,self.nbsf:]
        Xba = Px[self.nbsf:,:self.nbsf]
        Xbb = Px[self.nbsf:,self.nbsf:]
        Waa = Pw[:self.nbsf,:self.nbsf]
        Wab = Pw[:self.nbsf,self.nbsf:]
        Wba = Pw[self.nbsf:,:self.nbsf]
        Wbb = Pw[self.nbsf:,self.nbsf:]

        # Build the M matrix
        M = np.zeros((4,4)) 
        M[0,0] = np.real(np.trace(Xaa @ Waa) + np.trace(Xbb @ Wbb) + 2*np.trace(Xab @ Wba))
        M[0,1] = 2 * np.imag(np.trace(Xab @ Wba))
        M[0,2] = np.real(np.trace(Xaa @ Wab) + np.trace(Xab @ Wbb) - np.trace(Xba @ Waa) - np.trace(Xbb @ Wba))
        M[0,3] = np.imag(np.trace(Xaa @ Wab) + np.trace(Xab @ Wbb) + np.trace(Xba @ Waa) + np.trace(Xbb @ Wba))
        M[1,1] = np.real(np.trace(Xaa @ Waa) + np.trace(Xbb @ Wbb) - 2*np.trace(Xab @ Wba))
        M[1,2] = np.imag(np.trace(Xab @ Wbb) - np.trace(Xaa @ Wab) - np.trace(Xbb @ Wba) + np.trace(Xba @ Waa))
        M[1,3] = np.real(np.trace(Xaa @ Wab) - np.trace(Xab @ Wbb) + np.trace(Xba @ Waa) - np.trace(Xbb @ Wba))
        M[2,2] = np.real(np.trace(Xaa @ Wbb) + np.trace(Xbb @ Waa) - 2*np.trace(Xab @ Wab))
        M[2,3] = - 2 * np.imag(np.trace(Xab @ Wab))
        M[3,3] = np.real(np.trace(Xaa @ Wbb) + np.trace(Xbb @ Waa) + 2 * np.trace(Xab @ Wab))
        for i in range(4):
            for j in range(i):
                M[i,j] = M[j,i]
        
        # If we consider complex rotations, we need to compute the eigenvalues of M
        # Otherwise, we select only identity and Sy generators
        if(complex_rot):
            ev = np.linalg.eigvalsh(M)
        else:
            ev = np.linalg.eigvalsh(M[[0,2],:][:,[0,2]])
            
        return self.nelec - ev[-1]
    
    def excite(self,occ_idx,vir_idx):
        """ Perform orbital excitation on both spins
            Args:
                occ_idx : list of occupied orbital indices to be excited
                vir_idx : list of virtual orbital indices to be occupied
        """
        if(len(occ_idx)!=len(vir_idx)):
            raise ValueError("Occupied and virtual index lists must have the same length")
        source = occ_idx + vir_idx
        dest   = vir_idx + occ_idx
        self.mo_coeff[:,dest] = self.mo_coeff[:,source]
        self.update() 

    def mom_select(self, Cold, Cnew):
        """ Select new occupied orbital coefficients using MOM criteria 
            Args:
                Cold : Previous set of occupied orbital coefficients 
                Cnew : New set of orbital coefficients from Fock diagonalisation
            Returns:
                Cnew reordered according to MOM criterion
        """
        # Compute projections onto previous occupied space 
        p = np.einsum('pj,pq,ql->l', Cold[:,:self.nocc],self.ghf_overlap,Cnew,optimize="optimal")
        # Order MOs according to largest projection 
        idx = list(reversed(np.argsort(np.abs(p))))
        return Cnew[:,idx]

    def mo_cubegen(self,idx,fname=""): 
        """ Generate and store cube files for specified MOs
                idx : list of MO indices 
        """
        # Sum the alpha and beta spin AO contributions 
        mo_coeff = self.mo_coeff.copy()
        alpha = mo_coeff[:self.nbsf]
        beta = mo_coeff[self.nbsf:]
        spatial = alpha + beta
        # Saves MOs as cubegen files
        for mo in idx: 
            cubegen.orbital(self.integrals.mol, fname+f".mo.{mo}.cube", spatial[:,mo])

