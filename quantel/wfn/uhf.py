import scipy.linalg
import h5py
import numpy as np 
import quantel
from quantel.utils.scf_utils import mom_select
from quantel.utils.linalg import orthogonalise, matrix_print
from .wavefunction import Wavefunction
from pyscf.tools import cubegen 


class UHF(Wavefunction):
    """ Unrestricted Hartree-Fock method
    
        Inherits from the Wavefunction abstract base class with pure virtual properties:
            - energy
            - gradient
            - hessian 
            - take_step
            - save_last_step
            - restore_step
    """
    def __init__(self, integrals, verbose=0, mom_method=None): 
        """ Initialse UHF wavefunction """
        self.integrals = integrals
        self.nalfa = integrals.molecule().nalfa()  
        self.nbeta = integrals.molecule().nbeta()
       
        # Get number of basis functions and linearly independent orbitals 
        self.nbsf = integrals.nbsf()
        self.nmo = integrals.nmo()
        self.with_xc = (type(integrals) is not quantel.lib._quantel.LibintInterface)
        if(self.with_xc): self.with_xc = (integrals.xc is not None)

        self.verbose = verbose
        self.mom_method = mom_method
        self.nocc = (self.nalfa, self.nbeta)
        self.mo_occ = np.zeros((2,self.nmo))
        self.mo_occ[0,:self.nalfa] = 1.0
        self.mo_occ[1,:self.nbeta] = 1.0

        # Setup the indices for relevant spin orbital rotations
        self.rot_idx = self.uniq_var_indices() 
        self.nrot = (np.sum(self.rot_idx[0]), np.sum(self.rot_idx[1]))


    def initialise(self, mo_guess, ci_guess=None):
        """ Initialse the wave function with a set of molecalar orbital coefficients """
        if(len(mo_guess.shape) == 2):
            # We have 1 set of orbital coefficients so assume RHF guess
            self.mo_coeff = [mo_guess.copy(), mo_guess.copy()]
        elif(len(mo_guess.shape) == 3):
            # We have 2 sets of orbital coefficients
            self.mo_coeff = [mo_guess[0].copy(), mo_guess[1].copy()]
        else:
            raise ValueError("UHF initialisation requires 1 or 2 sets of molecular orbital coefficients")

        # Make sure orbitals are orthogonal
        self.mo_coeff[0] = orthogonalise(self.mo_coeff[0], self.integrals.overlap_matrix())
        self.mo_coeff[1] = orthogonalise(self.mo_coeff[1], self.integrals.overlap_matrix())

        # Update the density and Fock matrices
        self.update()

    @property
    def dim(self):
        """Get the number of degrees of freedom"""
        return self.nrot[0] + self.nrot[1]

    @property 
    def energy(self): 
        # Nuclear potential 
        En = self.integrals.scalar_potential()
        # One-electron energy
        h1e = self.integrals.oei_matrix()
        E1_alfa = np.einsum('pq,pq',h1e, self.dens[0], optimize="optimal")    
        E1_beta = np.einsum('pq,pq',h1e, self.dens[1], optimize="optimal")    
        # Two-electron energy 
        E2_alfa = 0.5 * np.einsum('pq,pq',self.JK[0],self.dens[0],optimize="optimal")
        E2_beta = 0.5 * np.einsum('pq,pq',self.JK[1],self.dens[1],optimize="optimal")
        # Exchange correlation 
        Exc = self.exc
        #Save components
        self.energy_components = dict()
        self.energy_components["Nuclear"]=En
        self.energy_components["One-Electron (alfa)"] = E1_alfa
        self.energy_components["One-Electron (beta)"]  = E1_beta
        self.energy_components["Two-Electron (alfa)"] = E2_alfa
        self.energy_components["Two-Electron (beta)"]  = E2_beta
        self.energy_components["Exchange_Correlation"] = Exc
        return En + E1_alfa + E1_beta + E2_alfa + E2_beta + Exc 

    @property
    def sz(self): 
        return 0.5*(self.nalfa - self.nbeta)

    @property
    def s2(self): 
        """ Get the total spin expectation value of the current UHF state """
        Ca_occ = self.mo_coeff[0][:,:self.nalfa]
        Cb_occ = self.mo_coeff[1][:,:self.nbeta]
        Sab = np.linalg.multi_dot([Ca_occ.T, self.integrals.overlap_matrix(), Cb_occ])
        return abs(self.sz*(self.sz+1)+self.nbeta - np.sum(Sab*Sab)) 

    @property
    def gradient(self):
        """ Energy gradient with respect to the spin orbital rotations """
        g_alfa = 2 * np.linalg.multi_dot([self.mo_coeff[0].T, self.fock[0], self.mo_coeff[0]])
        g_beta = 2 * np.linalg.multi_dot([self.mo_coeff[1].T, self.fock[1], self.mo_coeff[1]])
        grad = np.concatenate([g_alfa[self.rot_idx[0]],g_beta[self.rot_idx[1]]]) 
        return grad  

    @property
    def hessian(self):
        """Compute the internal UHF spin orbital Hessian"""
        # Number of occupied and virtual orbitals
        (no_a, no_b) = self.nocc
        nv_a = self.nmo - no_a
        nv_b = self.nmo - no_b

        # Compute Fock matrix in MO basis 
        Fmo_alfa = np.linalg.multi_dot([self.mo_coeff[0].T, self.fock[0], self.mo_coeff[0]])
        Fmo_beta = np.linalg.multi_dot([self.mo_coeff[1].T, self.fock[1], self.mo_coeff[1]])

        # Get occupied and virtual orbital coefficients
        Cocc_a = self.mo_coeff[0][:,:no_a].copy()
        Cvir_a = self.mo_coeff[0][:,no_a:].copy()
        Cocc_b = self.mo_coeff[1][:,:no_b].copy()
        Cvir_b = self.mo_coeff[1][:,no_b:].copy()

        # Compute ao_to_mo integral transform, all in physicists notation, middle values refer to the electron spins 
        eri_aa_rqsp = self.integrals.tei_ao_to_mo(Cvir_a,Cocc_a,Cocc_a,Cvir_a,True,False)
        eri_aa_rqps = self.integrals.tei_ao_to_mo(Cvir_a,Cocc_a,Cvir_a,Cocc_a,True,False)
        eri_aa_qsrp = self.integrals.tei_ao_to_mo(Cocc_a,Cocc_a,Cvir_a,Cvir_a,True,False)
        #
        eri_bb_rqsp = self.integrals.tei_ao_to_mo(Cvir_b,Cocc_b,Cocc_b,Cvir_b,True,False)
        eri_bb_rqps = self.integrals.tei_ao_to_mo(Cvir_b,Cocc_b,Cvir_b,Cocc_b,True,False)
        eri_bb_qsrp = self.integrals.tei_ao_to_mo(Cocc_b,Cocc_b,Cvir_b,Cvir_b,True,False)
        #
        eri_ab_rqsp = self.integrals.tei_ao_to_mo(Cvir_a,Cocc_b,Cocc_a,Cvir_b,True,False)
        
        # Construct Hessian matrix
        hess_aa = np.zeros((nv_a,no_a,nv_a,no_a))
        hess_bb = np.zeros((nv_b,no_b,nv_b,no_b))
        hess_ab = np.zeros((nv_a,no_a,nv_b,no_b))
        # Compute Fock contributions
        for i in range(no_a):
            hess_aa[:,i,:,i] += 2 * Fmo_alfa[no_a:,no_a:]
        for a in range(nv_a):
            hess_aa[a,:,a,:] -= 2 * Fmo_alfa[:no_a,:no_a]

        for i in range(no_b):
            hess_bb[:,i,:,i] += 2 * Fmo_beta[no_b:,no_b:]
        for a in range(nv_b):
            hess_bb[a,:,a,:] -= 2 * Fmo_beta[:no_b,:no_b]

        # Compute two-electron contributions
        # Alpha-Alpha terms 
        hess_aa += 4 * np.einsum('jabi->iajb', eri_aa_rqsp, optimize="optimal") 
        hess_aa -= 2 * self.integrals.hybrid_K * np.einsum('jaib->iajb', eri_aa_rqps, optimize="optimal") 
        hess_aa -= 2 * self.integrals.hybrid_K * np.einsum('abji->iajb', eri_aa_qsrp, optimize="optimal") 
        # Beta-Beta terms
        hess_bb += 4 * np.einsum('jabi->iajb', eri_bb_rqsp, optimize="optimal")
        hess_bb -= 2 * self.integrals.hybrid_K * np.einsum('jaib->iajb', eri_bb_rqps, optimize="optimal")
        hess_bb -= 2 * self.integrals.hybrid_K * np.einsum('abji->iajb', eri_bb_qsrp, optimize="optimal")        
        # Cross spin terms 
        hess_ab += 4 * np.einsum('rqsp->pqrs', eri_ab_rqsp, optimize="optimal")

        # Contribution from xc correlation
        if(not (self.integrals.xc is None)):
            # Build ground-state density and xc kernel
            rho0, vxc, fxc = self.integrals.cache_xc_kernel(self.mo_coeff,self.mo_occ,spin=1)

            # Loop over contributions per orbital pair
            for i in range(no_a):
                for a in range(nv_a):
                    # Build the first-order density matrix for this orbital pair
                    # These are weighted by the occupation difference
                    Dia = np.outer(Cocc_a[:,i], Cvir_a[:,a])
                    # Compute the contracted kernel with first-order density
                    fxc_ia = self.integrals.uks_fxc([Dia,np.zeros_like(Dia)],rho0,vxc,fxc)
                    # Compute contribution to Hessian diagonal
                    hess_aa[a,i,:,:] += 4 * np.linalg.multi_dot([Cvir_a.T, fxc_ia[0], Cocc_a])
                    hess_ab[a,i,:,:] += 2 * np.linalg.multi_dot([Cvir_b.T, fxc_ia[1], Cocc_b])
            for i in range(no_b):
                for a in range(nv_b):
                    # Build the first-order density matrix for this orbital pair
                    Dia = np.outer(Cocc_b[:,i], Cvir_b[:,a])
                    # Compute the contracted kernel with first-order density
                    fxc_ia = self.integrals.uks_fxc([np.zeros_like(Dia), Dia],rho0,vxc,fxc)
                    # Compute contribution to Hessian diagonal
                    hess_bb[a,i,:,:] += 4 * np.linalg.multi_dot([Cvir_b.T, fxc_ia[1], Cocc_b])
                    hess_ab[:,:,a,i] += 2 * np.linalg.multi_dot([Cvir_a.T, fxc_ia[0], Cocc_a])

        # Return suitably shaped array
        hess_aa = np.reshape(hess_aa, (nv_a*no_a, -1))
        hess_bb = np.reshape(hess_bb, (nv_b*no_b, -1))
        hess_ab = np.reshape(hess_ab, (nv_a*no_a, -1))
        return np.block([[hess_aa, hess_ab], [hess_ab.T, hess_bb]])



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
            print(f"        <S2> = {self.s2:5.2f}")
        if(verbose > 2):
            matrix_print(self.mo_coeff[0][:,:self.nalfa], title="Alpha Occupied Orbital Coefficients")
            matrix_print(self.mo_coeff[1][:,:self.nbeta], title="Beta Occupied Orbital Coefficients")
        if(verbose > 3):
            matrix_print(self.mo_coeff[0][:,self.nalfa:], title=" Alpha Virtual Orbital Coefficients", offset=self.nalfa)
            matrix_print(self.mo_coeff[1][:,self.nbeta:], title=" Beta Virtual Orbital Coefficients", offset=self.nalfa)
        if(verbose > 4):
            matrix_print(self.fock[0], title="Alpha Fock Matrix (AO basis)")
            matrix_print(self.fock[1], title="Beta Fock Matrix (AO basis)")
        print()
    
    def save_to_disk(self,tag):
        """Save object to disk with prefix 'tag'"""
        # Canonicalise orbitals
        self.canonicalize()
 
        # Save hdf5 file with MO coefficients, orbital energies, energy, and spin
        with h5py.File(tag+".hdf5", "w") as F:
            F.create_dataset("alpha mo_coeff", data=self.mo_coeff[0])
            F.create_dataset("beta mo_coeff" , data=self.mo_coeff[1])
            F.create_dataset("alpha mo_energy", data=self.mo_energy[0])
            F.create_dataset("beta mo_energy" , data=self.mo_energy[1])
            F.create_dataset("energy", data=self.energy)
            F.create_dataset("s2", data=self.s2)    

        # Save numpy txt file with energy and Hessian indices
        hindices = self.get_hessian_index()
        with open(tag+".solution", "w") as F:
            F.write(f"{self.energy:18.12f} {hindices[0]:5d} {hindices[1]:5d} {self.s2:12.6f}\n")
    
    def read_from_disk(self, tag):
        """Read a wavefunction object to disk"""
        with h5py.File(tag+".hdf5", "r") as F:
            mo_read = [F["alpha mo_coeff"][:], F["beta mo_coeff"][:]]
        self.initialise(mo_read)

    def update(self): 
        """ Update the density and Fock matrices for the current MO coefficients """
        self.get_density()
        self.get_fock()

    def get_density(self): 
        """ Construct the two alpha and beta density matrices for the current iteration """
        self.dens = np.zeros((2,self.nbsf,self.nbsf))
        self.dens[0] = self.mo_coeff[0][:,:self.nalfa] @ self.mo_coeff[0][:,:self.nalfa].T
        self.dens[1] = self.mo_coeff[1][:,:self.nbeta] @ self.mo_coeff[1][:,:self.nbeta].T

    def get_fock(self): 
        """ Construct the two alpha and beta Fock matrices for the current iteration """
        # Compute arrays of J and K matrices
        vJ, Ipqqp, vK = self.integrals.build_JK(self.dens,self.dens,Kxc=True)
        # Compute the exchange-correlation energy  
        self.exc , self.vxc = self.integrals.build_vxc(self.dens)
        # Compute JK contributions
        self.JK = (vJ[0]+vJ[1]-vK[0], vJ[0]+vJ[1]-vK[1])
        # Construct Fock matrices
        self.fock = np.zeros_like(self.dens)
        self.fock[0] = self.integrals.oei_matrix(True)  + self.JK[0] + self.vxc[0]
        self.fock[1] = self.integrals.oei_matrix(False) + self.JK[1] + self.vxc[1]
        # Vectorised format of the Fock matrices
        return np.reshape(self.fock,(-1))

    def canonicalize(self):
        """ Canonicalize the alpha and beta orbitals by diagonalising the occupied and virtual blocks of the Fock matrices """
        Q = np.zeros((2,self.nmo,self.nmo))
        self.mo_energy = np.zeros((2,self.nmo))
        for spin in range(2):
            # Get data for this spin sector
            nocc = self.nocc[spin]
            C = self.mo_coeff[spin]
            fock = self.fock[spin]
            # Get Fock matrix in MO basis
            Fmo = np.linalg.multi_dot([C.T, fock, C])
            # Extract occupied and virtual blocks
            Focc = Fmo[:nocc,:nocc]
            Fvir = Fmo[nocc:,nocc:]
            # Diagonalise the occupied and virtual blocks
            self.mo_energy[spin][:nocc], Qocc = np.linalg.eigh(Focc)
            self.mo_energy[spin][nocc:], Qvir = np.linalg.eigh(Fvir)
            # Build the canonical MO coefficients
            self.mo_coeff[spin][:,:nocc] = np.dot(C[:,:nocc], Qocc)     
            self.mo_coeff[spin][:,nocc:] = np.dot(C[:,nocc:], Qvir)
            # Combine full transformation matrix
            Q[spin][:nocc,:nocc] = Qocc
            Q[spin][nocc:,nocc:] = Qvir
        # Update the Fock matrices 
        self.update()
        return Q

    def get_preconditioner(self):
        """Compute approximate diagonal of Hessian"""
        # Get Fock matrix in MO basis 
        fock_mo = np.einsum('smp,smn,snq->spq', self.mo_coeff, self.fock, self.mo_coeff)
        # Initialise preconditioner
        Q = [np.zeros((self.nmo,self.nmo)), np.zeros((self.nmo,self.nmo))]
        # Include dominate Fock matrix terms
        for spin in range(2):
            for p in range(self.nmo):
                for q in range(self.nmo):
                    Q[spin][p,q] = 2 * (fock_mo[spin][p,p] - fock_mo[spin][q,q])

        return np.abs(np.concatenate((Q[0][self.rot_idx[0]], Q[1][self.rot_idx[1]])))

    def diagonalise_fock(self):
        """Diagonalise the Fock matrices via transformation of the generalised eigenvalue problem"""
        # Get the orthogonalisation matrix and overlap matrix
        X = self.integrals.orthogonalization_matrix()
        metric = self.integrals.overlap_matrix()
        # Project to linearly independent orbitals
        for spin in range(2):
            Ft = np.linalg.multi_dot([X.T, self.fock[spin], X])
            # Diagonalise the Fock matrix
            Et, Ct = np.linalg.eigh(Ft)
            # Transform back to the original basis
            Cnew = np.dot(X, Ct)
            # Apply MOM if required
            if(self.mom_method =='MOM'):
                Cold = self.mo_coeff[spin].copy()
                self.mo_coeff[spin] = mom_select(Cold[:,:self.nocc[spin]],Cnew,metric)
            elif(self.mom_method == 'IMOM'):
                self.mo_coeff[spin] = mom_select(self.Cinit[spin][:,:self.nocc[spin]],Cnew,metric)
            else:
                self.mo_coeff[spin] = Cnew.copy()
        # Update densities and Fock matrices
        self.update()

    def transform_vector(self, vec, step, X=None): 
        """ Perform orbital rotation for vector in tangent space"""
        # Extract spin components
        kappa = np.zeros((2,self.nmo,self.nmo))
        kappa[0][self.rot_idx[0]] = vec[:self.nrot[0]]
        kappa[1][self.rot_idx[1]] = vec[self.nrot[0]:]
        # Build antisymmetric matrices
        kappa = kappa - kappa.transpose((0,2,1))
        # transform as required
        if not X is None:
            for spin in range(2):
                kappa[spin] = kappa[spin] @ X[spin]
                kappa[spin] = X[spin].T @ kappa[spin]
        # Format new vector 
        vec_new = np.zeros_like(vec)
        vec_new[:self.nrot[0]] = kappa[0][self.rot_idx[0]]
        vec_new[self.nrot[0]:] = kappa[1][self.rot_idx[1]]
        return vec_new 

    def try_fock(self, fock_vec):
        """Try an extrapolated Fock matrix and update the orbital coefficients"""
        self.fock = fock_vec.reshape((2,self.nbsf,self.nbsf))
        self.diagonalise_fock()

    def get_diis_error(self): 
        """Compute DIIS error vector and DIIS error"""
        # Get the overlap matrix
        metric = self.integrals.overlap_matrix()
        # Initialise error vectors
        n2 = self.nbsf*self.nbsf
        err_vec = np.zeros((2*n2))
        for spin in range(2): 
            spin_err_vec  = np.linalg.multi_dot([self.fock[spin], self.dens[spin], metric])
            spin_err_vec -= spin_err_vec.T
            err_vec[spin*n2:(spin+1)*n2] = spin_err_vec.reshape((-1))    
        return err_vec, np.linalg.norm(err_vec)

    def restore_last_step(self): 
        """Restore orbital coefficients to the previous step"""
        self.mo_coeff = self.mo_coeff_save.copy()
        self.update()
    
    def save_last_step(self): 
        """Save current orbital coefficients"""
        self.mo_coeff_save = self.mo_coeff.copy()
    
    def take_step(self,step): 
        """Take a step in the orbital spaces"""
        self.save_last_step()
        self.rotate_orb(step) 
    
    def rotate_orb(self,step): 
        """Rotate MO coeffs with a step, step specifies our kpq coefficients"""
        step_a = step[:self.nrot[0]]
        step_b = step[self.nrot[0]:]
        # Build antisymm step matrices (antihermitian but also real) 
        Ka = np.zeros((self.nmo, self.nmo))
        Kb = np.zeros((self.nmo, self.nmo))
        Ka[self.rot_idx[0]]  = step_a
        Kb[self.rot_idx[1]]  = step_b
        # Build the Unitary transformations 
        Ua = scipy.linalg.expm(Ka - Ka.T)
        Ub = scipy.linalg.expm(Kb - Kb.T)
        # Transform the coefficients
        self.mo_coeff[0] = np.dot(self.mo_coeff[0], Ua) 
        self.mo_coeff[1] = np.dot(self.mo_coeff[1], Ub)
        # Update the density and the fock matrices
        self.update()
        
    def uniq_var_indices(self): 
        """ Creates matrices of boolean of size (nmo,nmo)
            Selects nonredudant rotations for the different spin channels """
        mask_alfa = np.zeros((self.nmo, self.nmo), dtype=bool)
        mask_beta = np.zeros((self.nmo, self.nmo), dtype=bool)
        # Includes only occupied-virtual spin-orbital rotations 
        mask_alfa[self.nocc[0]:,:self.nocc[0]] = True 
        mask_beta[self.nocc[1]:,:self.nocc[1]] = True 
        return mask_alfa, mask_beta 
    
    def get_orbital_guess(self, method="gwh",asymmetric=False):
        """ Get a guess for the molecular orbital coefficients 
            Asymmetric initial guess can be generated via HOMO-LUMO mixing"""
        # Get one-electron integrals and overlap matrix 
        h1e = self.integrals.oei_matrix(True)
        s = self.integrals.overlap_matrix()
        
        # Build guess Fock matrix
        if(method.lower() == "core"):
            # Use core Hamiltonian as guess
            self.fock = [h1e.copy(), h1e.copy()]
        elif(method.lower() == "gwh"):
            # Build GWH guess Hamiltonian
            K = 1.75
            fock0 = np.zeros((self.nbsf,self.nbsf))
            for i in range(self.nbsf):
                for j in range(self.nbsf):
                    fock0[i,j] = 0.5 * (h1e[i,i] + h1e[j,j]) * s[i,j]
                    if(i!=j):
                        fock0[i,j] *= 1.75
            self.fock = [fock0.copy(), fock0.copy()]
        else:
            raise NotImplementedError(f"Orbital guess method {method} not implemented")
        
        # Get orbital coefficients by diagonalising Fock matrix
        # We do this explicitly here to get around IMOM condition
        X = self.integrals.orthogonalization_matrix()
        Cinit = np.zeros((2,self.nbsf,self.nbsf))
        for spin in range(2):
            Ft = np.linalg.multi_dot([X.T, self.fock[spin], X])
            Et, Ct = np.linalg.eigh(Ft)    
            Cinit[spin] = np.dot(X, Ct)
        # Get orbital coefficients by diagonalising Fock matrix
        self.initialise(Cinit)
        return

        # Compute an asymmetric initial guess
        # HGAB: 22/12/2025
        #       Maybe we should write a new driver routine to perform this type of routine?
        if asymmetric:
            k = 0.1
            # Extract the previous HOMO and LUMO MO coefficients
            mo_coeff = self.beta.mo_coeff.copy()
            homo = mo_coeff[:, self.beta.nocc-1] 
            lumo = mo_coeff[:, self.beta.nocc] 
            # Mix HOMO and LUMO 
            pos_mix = (1/np.sqrt(1+k**2)) * (homo + k*lumo)
            neg_mix = (1/np.sqrt(1+k**2)) * (-k*homo + lumo)
            mo_coeff[:, self.beta.nocc-1] = pos_mix
            mo_coeff[:, self.beta.nocc] = neg_mix
            self.beta.mo_coeff = mo_coeff
            self.update()
   
    def excite(self): 
        """ Performs an Sz preserving HOMO LUMO excitation """
        # HGAB: 22/12/2025
        #       Maybe we should write a new driver routine to perform this type of routine?
        alfa_gap =  self.alfa.mo_energy[self.alfa.nocc]-self.alfa.mo_energy[self.alfa.nocc-1]
        beta_gap =  self.beta.mo_energy[self.beta.nocc]-self.beta.mo_energy[self.beta.nocc-1]
        if(alfa_gap < beta_gap): 
            self.alfa.mo_coeff[:,[self.alfa.nocc -1, self.alfa.nocc]] = self.alfa.mo_coeff[:,[self.alfa.nocc, self.alfa.nocc-1]]
        else: 
            self.beta.mo_coeff[:,[self.beta.nocc -1, self.beta.nocc]] = self.beta.mo_coeff[:,[self.beta.nocc, self.beta.nocc-1]]
        # Update the density and Fock matrices
        self.update()
 
    @property 
    def value(self):
        """Map the energy onto the function value"""
        return self.energy

    def overlap(self,other):
        return 

    def hamiltonian(self, other):
        """Compute the Hamiltonian coupling with another wavefunction of this type"""
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
    

    def hess_on_vec(self,X):
        """ Compute the direct action of Hessian on a vector X (much faster than building the full Hessian)"""
        # Number of occupied and virtual orbitals
        (no_a, no_b) = self.nocc
        (nv_a, nv_b) = (self.nmo - no_a, self.nmo - no_b)
        # Split vector into alpha and beta parts
        Xai_alfa = np.reshape(X[:self.nrot[0]], (nv_a, no_a))
        Xai_beta = np.reshape(X[self.nrot[0]:], (nv_b, no_b))
        # Access occupied and virtual orbital coefficients
        Ci_alfa = self.mo_coeff[0][:,:no_a].copy()
        Ca_alfa = self.mo_coeff[0][:,no_a:].copy()
        Ci_beta = self.mo_coeff[1][:,:no_b].copy()
        Ca_beta = self.mo_coeff[1][:,no_b:].copy()
    
        # First order density change
        D1a = np.einsum('pa,ai,qi->pq', Ca_alfa, Xai_alfa, Ci_alfa, optimize="optimal")
        D1b = np.einsum('pa,ai,qi->pq', Ca_beta, Xai_beta, Ci_beta, optimize="optimal")
        # Coulomb and exchange contributions
        J, K = self.integrals.build_JK([D1a,D1b],[D1a,D1b], Kxc=False)
        # Build ground-state density and xc kernel
        if(not (self.integrals.xc is None)):
            rho0, vxc, fxc = self.integrals.cache_xc_kernel(self.mo_coeff,self.mo_occ,spin=1)
            fxc = self.integrals.uks_fxc([D1a,D1b],rho0,vxc,fxc)
        else:
            fxc = np.zeros_like(J)
        
        # Fock contributions
        Fba_alfa = np.linalg.multi_dot([Ca_alfa.T, self.fock[0], Ca_alfa])
        Fij_alfa = np.linalg.multi_dot([Ci_alfa.T, self.fock[0], Ci_alfa])
        Fba_beta = np.linalg.multi_dot([Ca_beta.T, self.fock[1], Ca_beta])
        Fij_beta = np.linalg.multi_dot([Ci_beta.T, self.fock[1], Ci_beta])
        
        # Initialise output 
        HX = np.zeros_like(X)
        HX[:self.nrot[0]] = 2 * (Fba_alfa @ Xai_alfa - Xai_alfa @ Fij_alfa).ravel()
        HX[self.nrot[0]:] = 2 * (Fba_beta @ Xai_beta - Xai_beta @ Fij_beta).ravel()
        
        # Compute 2-electron contributions
        kernel_a = 4 * (J[0] + J[1]) + (4*fxc[0]) - 2 * self.integrals.hybrid_K * (K[0] + K[0].T)
        kernel_b = 4 * (J[0] + J[1]) + (4*fxc[1]) - 2 * self.integrals.hybrid_K * (K[1] + K[1].T)
        HX[:self.nrot[0]] += np.linalg.multi_dot([Ca_alfa.T, kernel_a, Ci_alfa]).ravel()
        HX[self.nrot[0]:] += np.linalg.multi_dot([Ca_beta.T, kernel_b, Ci_beta]).ravel()
        return HX
    
    def mo_cubegen(self, a_idx, b_idx, fname=""): 
        """ Generate and store cube files for specified MOs
                a_idx, b_idx : lists of indices of alpha and beta MOs
        """
        spins = ["a","b"]
        for i, spin in enumerate([self.alfa, self.beta]):    
            for mo in [a_idx,b_idx][i]: 
                cubegen.orbital(self.integrals.mol, fname+f".{spins[i]}.mo.{mo}.cube", spin.mo_coeff[:,mo])

