#!/usr/bin/python3
# Author: Hugh G. A. Burton

import numpy as np
import scipy.linalg
import h5py
from quantel.utils.linalg import orthogonalise, matrix_print
from quantel.utils.scf_utils import mom_select
from .wavefunction import Wavefunction
import quantel
from pyscf.tools import cubegen

class RHF(Wavefunction):
    """ Restricted Hartree-Fock method

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
        self.nalfa     = integrals.molecule().nalfa()
        self.nbeta     = self.nalfa

        # Get number of basis functions and linearly independent orbitals
        self.nbsf      = integrals.nbsf()
        self.nmo       = integrals.nmo()
        self.with_xc    = (type(integrals) is not quantel.lib._quantel.LibintInterface)
        if(self.with_xc): self.with_xc = (integrals.xc is not None)

        # For now, we assume that the number of alpha and beta electrons are the same
        assert(self.nalfa == self.nbeta)
        self.nocc      = self.nalfa
        self.verbose   = verbose
        self.mom_method = mom_method

        # Setup the indices for relevant orbital rotations
        self.rot_idx   = self.uniq_var_indices() 
        self.nrot      = np.sum(self.rot_idx) 

        # Define the orbital energies and coefficients
        self.mo_coeff  = None
        self.mo_energy = None
    
    def initialise(self, mo_guess, ci_guess=None):
        """Initialise the wave function with a set of molecular orbital coefficients"""
        # Make sure orbitals are orthogonal
        self.mo_coeff = orthogonalise(mo_guess, self.integrals.overlap_matrix())
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
        # Nuclear potential
        En  = self.integrals.scalar_potential()
        # One-electron energy
        E1 = 2 * np.einsum('pq,pq', self.integrals.oei_matrix(True), self.dens, optimize="optimal")
        # Two-electron energy
        E2 = np.einsum('pq,pq', self.JK, self.dens, optimize="optimal")
        # Exchange correlation
        Exc = self.exc
        # Save components
        self.energy_components = dict(Nuclear=En, One_Electron=E1, Two_Electron=E2, Exchange_Correlation=Exc)
        return En + E1 + E2 + Exc

    @property
    def s2(self):
        """Get the spin of the current RHF state"""
        return 0 # All RHF states have spin 0

    @property
    def gradient(self):
        """Get the energy gradient with respect to the orbital rotations"""
        g = 4 * np.linalg.multi_dot([self.mo_coeff.T, self.fock, self.mo_coeff])
        return g[self.rot_idx]

    @property
    def hessian(self):
        """Compute the internal RHF orbital Hessian"""
        # Number of occupied and virtual orbitals
        no = self.nocc
        nv = self.nmo - self.nocc

        # Compute Fock matrix in MO basis 
        Fmo = np.linalg.multi_dot([self.mo_coeff.T, self.fock, self.mo_coeff])

        # Get two-electron integrals if not already computed
        if(not hasattr(self, 'eri_abij') or not hasattr(self, 'eri_aibj')):
            self.update(with_eri=True)

        # Initialise Hessian matrix
        hessian = np.zeros((nv,no,nv,no))

        # Compute Fock contributions
        for i in range(no):
            hessian[:,i,:,i] += 4 * Fmo[no:,no:]
        for a in range(nv):
            hessian[a,:,a,:] -= 4 * Fmo[:no,:no]

        # Compute two-electron contributions
        hessian += 16 * np.einsum('abij->aibj', self.eri_abij, optimize="optimal")
        hessian -=  4 * self.integrals.hybrid_K * np.einsum('ajbi->aibj', self.eri_aibj, optimize="optimal")
        hessian -=  4 * self.integrals.hybrid_K * np.einsum('abji->aibj', self.eri_abij, optimize="optimal")

        if(not (self.integrals.xc is None)):
            # Build ground-state density and xc kernel
            occ = np.zeros(self.nmo)
            occ[:self.nocc] = 1.0
            rho0, vxc, fxc = self.integrals.cache_xc_kernel([self.mo_coeff,self.mo_coeff],(occ,occ),spin=1)

            # Loop over contributions per orbital pair
            for i in range(no):
                for a in range(nv):
                    # Build the first-order density matrix for this orbital pair
                    # These are weighted by the occupation difference
                    Dia = np.outer(self.mo_coeff[:,i],self.mo_coeff[:,no+a])
                    # Compute the contracted kernel with first-order density
                    fxc_ia = self.integrals.uks_fxc(Dia,rho0,vxc,fxc)
                    # Compute contribution to Hessian diagonal
                    hessian[a,i,:,:] += 16 * self.mo_coeff[:,self.nocc:].T @ (fxc_ia[0] @ self.mo_coeff[:,:self.nocc])

        # Return suitably shaped array
        return np.reshape(hessian, (nv*no,-1))
    

    def hess_on_vec(self,X):
        """ Compute the action of Hessian on a vector X"""
        # Reshape X into matrix form
        Xai = np.reshape(X, (self.nmo-self.nocc,self.nocc))
        # Access occupied and virtual orbitals
        Ci = self.mo_coeff[:,:self.nocc]
        Ca = self.mo_coeff[:,self.nocc:]

        # First order density change
        Dia = np.einsum('pa,ai,qi->pq', Ca, Xai, Ci, optimize="optimal")
        # Coulomb and exchange contributions
        Jia, Kia, = self.integrals.build_JK([Dia],[Dia],hermi=0,Kxc=False)
        # Build ground-state density and fxc kernel
        if(not (self.integrals.xc is None)):
            occ = np.zeros(self.nmo)
            occ[:self.nocc] = 1.0
            rho0, vxc, fxc = self.integrals.cache_xc_kernel([self.mo_coeff,self.mo_coeff],(occ,occ),spin=1)
            fxc = self.integrals.uks_fxc(Dia, rho0, vxc, fxc)[0]
        else:
            fxc = np.zeros_like(Jia[0])
        
        # Fock contributions 
        Fba = Ca.T @ self.fock @ Ca
        Fij = Ci.T @ self.fock @ Ci
        HX = 4 * (Fba @ Xai - Xai @ Fij)
        
        # Compute contribution to hess_on_vec
        kernel = 16 * (Jia[0] + fxc) - 4 * self.integrals.hybrid_K * (Kia[0] + Kia[0].T)
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
            print(f"        <Sz> = {0:5.2f}")
            print(f"        <S2> = {self.s2:5.2f}")
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
        them = RHF(self.integrals, verbose=self.verbose)
        them.initialise(self.mo_coeff)
        return them

    def overlap(self, them):
        """Compute the (nonorthogonal) many-body overlap with another RHF wavefunction (them)"""
        if(self.nocc != them.nocc):
            return 0
        nocc = self.nocc
        ovlp = self.integrals.overlap_matrix()
        S = np.linalg.multi_dot([self.mo_coeff[:,:nocc].T, ovlp, them.mo_coeff[:,:nocc]])
        return np.linalg.det(S)**2

    def hamiltonian(self, them):
        """Compute the (nonorthogonal) many-body Hamiltonian coupling with another RHF wavefunction (them)"""
        raise NotImplementedError("RHF Hamiltonian not implemented")

    def update(self, with_eri=True):
        """Update the 1RDM and Fock matrix for the current state"""
        self.get_density()
        self.get_fock()
        if(with_eri):
            # Get occupied and virtual orbital coefficients
            Cocc = self.mo_coeff[:,:self.nocc].copy()
            Cvir = self.mo_coeff[:,self.nocc:].copy()
            # Compute ao_to_mo integral transform
            self.eri_abij = self.integrals.tei_ao_to_mo(Cvir,Cvir,Cocc,Cocc,True,False)
            self.eri_aibj = self.integrals.tei_ao_to_mo(Cvir,Cocc,Cvir,Cocc,True,False)

    def get_density(self):
        """Compute the 1RDM for the current state in AO basis"""
        Cocc = self.mo_coeff[:,:self.nocc]
        self.dens = np.dot(Cocc, Cocc.T)

    def get_fock(self):
        """Compute the Fock matrix for the current state"""
        # Compute the Coulomb and Exchange matrices
        J, self.Ipqqp, K = self.integrals.build_JK([self.dens],[self.dens],hermi=1,Kxc=True)
        self.JK = 2*J[0] - K[0]
        # Compute the exchange-correlation energy
        self.exc, self.vxc = self.integrals.build_vxc([self.dens, self.dens])
        self.fock = self.integrals.oei_matrix(True) + self.JK + self.vxc[0]
        return self.fock.reshape((-1))

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
                Q[p,q] = 4 * (fock_mo[p,p] - fock_mo[q,q])
        return np.abs(Q[self.rot_idx])

    def diagonalise_fock(self):
        """Diagonalise the Fock matrix"""
        # Get the orthogonalisation matrix
        X = self.integrals.orthogonalization_matrix()
        # Project to linearly independent orbitals
        Ft = np.linalg.multi_dot([X.T, self.fock, X])
        # Diagonalise the Fock matrix
        Et, Ct = np.linalg.eigh(Ft)
        # Transform back to the original basis
        Cnew = np.dot(X, Ct)

        # Select occupied orbitals using MOM if specified
        if(self.mom_method =='MOM'):
            Cold = self.mo_coeff.copy()
            self.mo_coeff = mom_select(Cold[:,:self.nocc],Cnew,self.integrals.overlap_matrix())
        elif(self.mom_method == 'IMOM'):
            self.mo_coeff = mom_select(self.Cinit[:,:self.nocc],Cnew,self.integrals.overlap_matrix())
        else:
            self.mo_coeff = Cnew.copy()

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
        self.fock = fock_vec.reshape((self.nbsf,self.nbsf)).T
        self.diagonalise_fock()

    def get_diis_error(self):
        """Compute the DIIS error vector and DIIS error"""
        err_vec  = np.linalg.multi_dot([self.fock, self.dens, self.integrals.overlap_matrix()])
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
            self.fock = h1e.copy()
        elif(method.lower() == "gwh"):
            # Build GWH guess Hamiltonian
            K = 1.75
            self.fock = np.zeros((self.nbsf,self.nbsf))
            for i in range(self.nbsf):
                for j in range(self.nbsf):
                    self.fock[i,j] = 0.5 * (h1e[i,i] + h1e[j,j]) * s[i,j]
                    if(i!=j):
                        self.fock[i,j] *= 1.75
            
        else:
            raise NotImplementedError(f"Orbital guess method {method} not implemented")
        
        # Get the orthogonalisation matrix
        X = self.integrals.orthogonalization_matrix()
        # Project to linearly independent orbitals
        Ft = np.linalg.multi_dot([X.T, self.fock, X])
        # Diagonalise the Fock matrix
        Et, Ct = np.linalg.eigh(Ft)    
        Cinit = np.dot(X, Ct)
        # Get orbital coefficients by diagonalising Fock matrix
        self.initialise(Cinit)

    def excite(self,occ_idx,vir_idx,mom_method=None):
        """ Perform orbital excitation on both spins
            Args:
                occ_idx : list of occupied orbital indices to be excited
                vir_idx : list of virtual orbital indices to be occupied
        """
        if(len(occ_idx)!=len(vir_idx)):
            raise ValueError("Occupied and virtual index lists must have the same length")
        source = occ_idx + vir_idx
        dest   = vir_idx + occ_idx
        coeff_new = self.mo_coeff.copy()
        coeff_new[:,dest] = self.mo_coeff[:,source]
        them = RHF(self.integrals,verbose=self.verbose,mom_method=mom_method)
        them.initialise(coeff_new)
        return them

    def deallocate(self):
        pass
        
    def approx_hess_on_vec(self, vec, eps=1e-3):
        """ Compute the approximate Hess * vec product using forward finite difference """
        # Get current gradient
        g0 = self.gradient.copy()
        # Save current position
        self.save_last_step()
        # Get forward gradient
        self.take_step(eps * vec)
        g1 = self.gradient.copy()
        # Restore to origin
        self.restore_last_step()
        # Parallel transport back to current position
        g1 = self.transform_vector(g1, - eps * vec)
        # Get approximation to H @ sk
        return (g1 - g0) / eps

    def mo_cubegen(self,idx=None,fname=""): 
        """ Generate and store cube files for specified MOs
                idx : list of MO indices 
        """
        if(idx is None): 
            idx = range(self.nmo)
        # Saves MOs as cubegen files
        for mo in idx: 
            cubegen.orbital(self.integrals.mol, fname+f".mo.{mo}.cube", self.mo_coeff[:,mo])