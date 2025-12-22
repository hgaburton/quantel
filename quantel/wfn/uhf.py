import scipy.linalg
import h5py
import numpy as np 
import quantel
from .wavefunction import Wavefunction
from pyscf.tools import cubegen 

class SpinChannel:
    """UHF helper class to contain spin-specific wavefunction properties."""
    def __init__(self):
        self.nocc = None
        self.mo_occ = None
        self.fock = None
        # Define the orbital energies and coefficients 
        self.mo_coeff = None
        self.mo_coeff_save = None
        self.mo_energy = None
        # Indices for orbital rotations
        self.rot_idx = None
        self.nrot = None

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
    def __init__(self, integrals, verbose=0): 
        """ Initialse UHF wavefunction """
        self.integrals = integrals
        self.nalfa = integrals.molecule().nalfa()  
        self.nbeta = integrals.molecule().nbeta()
       
        # Get number of basis functions and linearly independent orbitals 
        self.nbsf = integrals.nbsf()
        self.nmo = integrals.nmo()
        self.with_xc = (type(integrals) is not quantel.lib._quantel.LibintInterface)
        if(self.with_xc): self.with_xc = (integrals.xc is not None)

        #Set up hierarchical structure 
        self.alfa = SpinChannel()
        self.beta = SpinChannel()
        
        # Define spin occupancies
        self.alfa.nocc = self.nalfa
        self.beta.nocc = self.nbeta 
        
        # Setup the indices for relevant spin orbital rotations
        """ Spin specific information """
        self.alfa.rot_idx, self.beta.rot_idx   = self.uniq_var_indices() 
        self.alfa.nrot = np.sum(self.alfa.rot_idx)
        self.beta.nrot = np.sum(self.beta.rot_idx)
        """ Total rotational DOFs over both spin channels"""
        self.nrot = np.sum(self.alfa.rot_idx) + np.sum(self.alfa.rot_idx)
        """ Overall matrix of nonredudant rotations """   
        self.rot_idx = np.zeros((2*self.nmo, 2*self.nmo), dtype=bool) 
        self.rot_idx[:self.nmo, :self.nmo] = self.alfa.rot_idx
        self.rot_idx[self.nmo:, self.nmo:] = self.beta.rot_idx

    def initialise(self, mo_guess, ci_guess=None):
        """ Initialse the wave function with a set of molecalar orbital coefficients """
        # Make sure orbitals are orthogonal
        self.alfa.mo_coeff = orthogonalise(mo_guess, self.integrals.overlap_matrix())
        self.beta.mo_coeff = orthogonalise(mo_guess, self.integrals.overlap_matrix())
        # Update the density and Fock matrices
        self.update()

    @property
    def dim(self):
        """Get the number of degrees of freedom"""
        return self.nrot

    @property
    def mo_coeff(self):
        """ Get list of MO coeffs"""
        return [self.alfa.mo_coeff, self.beta.mo_coeff]

    @property 
    def energy(self): 
        # Nuclear potential 
        En = self.integrals.scalar_potential()
        # One-electron energy
        E1_alfa = np.einsum('pq,pq',self.integrals.oei_matrix(True), self.alfa.dens, optimize="optimal")    
        E1_beta = np.einsum('pq,pq',self.integrals.oei_matrix(True), self.beta.dens, optimize="optimal")    
        # Two-electron energy 
        E2_alfa = 0.5*np.einsum('pq,pq',self.alfa.JK, self.alfa.dens, optimize="optimal")
        E2_beta = 0.5*np.einsum('pq,pq',self.beta.JK, self.beta.dens, optimize="optimal")
        # Exchange correlation 
        Exc = self.exc   
        #Save components
        self.energy_components = dict(Nuclear=En, One_Electron_alfa=E1_alfa, One_Electron_beta=E1_beta, Two_Electron_alfa=E2_alfa, Two_Electron_beta=E2_beta, Exchange_Correlation=Exc )
        return En + E1_alfa + E1_beta + E2_alfa + E2_beta + Exc 

    @property
    def sz(self): 
        return 0.5*(self.nalfa - self.nbeta)

    @property
    def s2(self): 
        """ Get the total spin expectation value of the current UHF state """
        ab_overlap = np.linalg.multi_dot(((self.alfa.mo_coeff[:,:self.alfa.nocc]).T,self.integrals.overlap_matrix(), self.beta.mo_coeff[:,:self.beta.nocc]))
        return abs(self.sz*( self.sz + 1 ) + self.nbeta - np.sum( np.abs(ab_overlap)**2 )) 

    @property
    def gradient(self):
        """ Energy gradient with respect to the spin orbital rotations """
        g_alfa = 2 * np.linalg.multi_dot([self.alfa.mo_coeff.T, self.alfa.fock, self.alfa.mo_coeff])
        g_beta = 2 * np.linalg.multi_dot([self.beta.mo_coeff.T, self.beta.fock, self.beta.mo_coeff])
        g_alfa = g_alfa[self.alfa.rot_idx]
        g_beta = g_beta[self.beta.rot_idx]
        # Overall gradient
        grad = np.array([*g_alfa, *g_beta]) 
        return grad  

    @property
    def hessian(self):
        """Compute the internal UHF spin orbital Hessian"""
        # Number of occupied and virtual orbitals
        no_a = self.alfa.nocc
        nv_a = self.nmo - self.alfa.nocc
        no_b = self.beta.nocc
        nv_b = self.nmo - self.beta.nocc

        # Compute Fock matrix in MO basis 
        Fmo_alfa = np.linalg.multi_dot([self.alfa.mo_coeff.T, self.alfa.fock, self.alfa.mo_coeff])
        Fmo_beta = np.linalg.multi_dot([self.beta.mo_coeff.T, self.beta.fock, self.beta.mo_coeff])

        # Get occupied and virtual orbital coefficients
        Cocc_a = self.alfa.mo_coeff[:,:no_a].copy()
        Cvir_a = self.alfa.mo_coeff[:,no_a:].copy()
        Cocc_b = self.beta.mo_coeff[:,:no_b].copy()
        Cvir_b = self.beta.mo_coeff[:,no_b:].copy()

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
        eri_ba_rqsp = self.integrals.tei_ao_to_mo(Cvir_b,Cocc_a,Cocc_b,Cvir_a,True,False)
        
        # Construct Hessian matrix
        hessian = np.zeros((2*self.nmo,2*self.nmo,2*self.nmo,2*self.nmo))
        # Compute Fock contributions
        for i in range(no_a):
            hessian[no_a:self.nmo,i,no_a:self.nmo,i] += 2 * Fmo_alfa[no_a:,no_a:]
        for a in range(no_a,self.nmo):
            hessian[a,:no_a,a,:no_a] -= 2 * Fmo_alfa[:no_a,:no_a]

        for i in range(self.nmo, no_b):
            hessian[ (self.nmo+no_b):, i, (self.nmo+no_b):,i] += 2 * Fmo_beta[no_b:,no_b:]
        for a in range(self.nmo+no_b,2*self.nmo):
            hessian[ a, self.nmo:(self.nmo+no_b), a, self.nmo:(self.nmo + no_b)] -= 2 * Fmo_beta[:no_b,:no_b]

        # Compute two-electron contributions
        # Alpha-Alpha terms 
        hessian[ no_a:self.nmo,:no_a, no_a:self.nmo,:no_a] += 4 * np.einsum('rqsp->pqrs', eri_aa_rqsp, optimize="optimal") 
        hessian[ no_a:self.nmo,:no_a, no_a:self.nmo,:no_a] -= 2 * np.einsum('rqps->pqrs', eri_aa_rqps, optimize="optimal") 
        hessian[ no_a:self.nmo,:no_a, no_a:self.nmo,:no_a] -= 2 * np.einsum('qsrp->pqrs', eri_aa_qsrp, optimize="optimal") 
        # Beta-Beta terms
        hessian[ (self.nmo+no_b):,self.nmo: (no_b+self.nmo), (self.nmo+no_b):,self.nmo:(no_b+ self.nmo)] += 4 * np.einsum('rqsp->pqrs', eri_bb_rqsp, optimize="optimal") 
        hessian[ (self.nmo+no_b):,self.nmo: (no_b+self.nmo), (self.nmo+no_b):,self.nmo:(no_b+ self.nmo)] -= 2 * np.einsum('rqps->pqrs', eri_bb_rqps, optimize="optimal") 
        hessian[ (self.nmo+no_b):,self.nmo: (no_b+self.nmo), (self.nmo+no_b):,self.nmo:(no_b+ self.nmo)] -= 2 * np.einsum('qsrp->pqrs', eri_bb_qsrp, optimize="optimal") 
        # Cross spin terms 
        hessian[ no_a:self.nmo, :no_a, (self.nmo+no_b):, self.nmo:(no_b+self.nmo)] += 4 * np.einsum('rqsp->pqrs', eri_ba_rqsp, optimize="optimal")     
        hessian[ (self.nmo+no_b):, self.nmo:(no_b+self.nmo), no_a:self.nmo, :no_a] += 4 * np.einsum('rqsp->pqrs', eri_ab_rqsp, optimize="optimal")      
        # Return suitably shaped array
        return (hessian[:,:, self.rot_idx])[self.rot_idx, :] 
    
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
            matrix_print(self.alfa.mo_coeff[:,:self.alfa.nocc], title="Alpha Occupied Orbital Coefficients")
            matrix_print(self.beta.mo_coeff[:,:self.beta.nocc], title="Beta Occupied Orbital Coefficients")
        if(verbose > 3):
            matrix_print(self.alfa.mo_coeff[:,self.alfa.nocc:], title=" Alpha Virtual Orbital Coefficients", offset=self.alfa.nocc)
            matrix_print(self.beta.mo_coeff[:,self.beta.nocc:], title=" Beta Virtual Orbital Coefficients", offset=self.alfa.nocc)
        if(verbose > 4):
            matrix_print(self.alfa.fock, title="Alpha Fock Matrix (AO basis)")
            matrix_print(self.beta.fock, title="Beta Fock Matrix (AO basis)")
        print()
    
    def save_to_disk(self,tag):
        """Save object to disk with prefix 'tag'"""
        # Canonicalise orbitals
        self.canonicalize()
 
        # Save hdf5 file with MO coefficients, orbital energies, energy, and spin
        with h5py.File(tag+".hdf5", "w") as F:
            F.create_dataset("alpha mo_coeff", data=self.alfa.mo_coeff)
            F.create_dataset("beta mo_coeff" , data=self.beta.mo_coeff)
            F.create_dataset("alpha mo_energy", data=self.alfa.mo_energy)
            F.create_dataset("beta mo_energy" , data=self.beta.mo_energy)
            F.create_dataset("energy", data=self.energy)
            F.create_dataset("s2", data=self.s2)    
        print("Are we going through here?", flush=True) 
        # Save numpy txt file with energy and Hessian indices
        hindices = self.get_hessian_index()
        with open(tag+".solution", "w") as F:
            F.write(f"{self.energy:18.12f} {hindices[0]:5d} {hindices[1]:5d} {self.s2:12.6f}\n")


    def update(self): 
        self.get_density()
        self.get_fock()

    def get_density(self): 
        alfa_Cocc = self.alfa.mo_coeff[:, :self.alfa.nocc] 
        beta_Cocc = self.beta.mo_coeff[:, :self.beta.nocc]
        self.alfa.dens = np.dot(alfa_Cocc, alfa_Cocc.T) #OE dens in AO basis. 
        self.beta.dens = np.dot(beta_Cocc, beta_Cocc.T)

    def get_fock(self): 
        """ Construct the two alpha and beta Fock matrices for the current iteration """
        # Define an array of AO density matrices 
        vdK = np.zeros((2,self.nbsf, self.nbsf))
        vdK[0,:,:]= self.alfa.dens
        vdK[1,:,:]= self.beta.dens
        # Compute arrays of J and K matrices
        vJ, vK = self.integrals.build_JK(vdK, vdK)
        # Extract alpha and beta terms 
        self.alfa.JK = vJ[0] + vJ[1] - vK[0]
        self.beta.JK = vJ[0] + vJ[1] - vK[1]
        # Construct Fock matrices
        self.alfa.fock = self.integrals.oei_matrix(True) + self.alfa.JK 
        self.beta.fock = self.integrals.oei_matrix(True) + self.beta.JK 
        # Compute the exchange-correlation energy  
        self.exc , self.alfa.vxc, self.beta.vxc = self.integrals.build_vxc(self.alfa.dens, self.beta.dens) if(self.with_xc) else 0,0,0 
        # Exchange correlation correction
        self.alfa.fock += self.alfa.vxc
        self.beta.fock += self.beta.vxc
        # Vectorised format of the Fock matrices
        self.fock_vec = np.concatenate(( self.alfa.fock.T.reshape((-1)) , self.beta.fock.T.reshape((-1))  ))

    def canonicalize(self):
        Qs = []
        for spin in [self.alfa, self.beta]:
            # Initialise orbital energies
            spin.mo_energy = np.zeros(self.nmo)
            # Get Fock matrix in MO basis
            Fmo = np.linalg.multi_dot([spin.mo_coeff.T, spin.fock, spin.mo_coeff])
            # Extract occupied and virtual blocks
            Focc = Fmo[:spin.nocc,:spin.nocc]
            Fvir = Fmo[spin.nocc:,spin.nocc:]
            # Diagonalise the occupied and virtual blocks
            spin.mo_energy[:spin.nocc], Qocc = np.linalg.eigh(Focc)
            spin.mo_energy[spin.nocc:], Qvir = np.linalg.eigh(Fvir)
            # Build the canonical MO coefficients
            spin.mo_coeff[:,:spin.nocc] = np.dot(spin.mo_coeff[:,:spin.nocc], Qocc)     
            spin.mo_coeff[:,spin.nocc:] = np.dot(spin.mo_coeff[:,spin.nocc:], Qvir)
            # Get orbital occupation
            spin.mo_occ = np.zeros(self.nmo)
            spin.mo_occ[:spin.nocc] = 1.0 
            # Combine full transformation matrix
            Q = np.zeros((self.nmo,self.nmo))
            Q[:spin.nocc,:spin.nocc] = Qocc
            Q[spin.nocc:,spin.nocc:] = Qvir
            Qs.append(Q)
        # Update the Fock matrices 
        self.update()
        # Construct the full transformation matrix: 
        Q_full = np.zeros((2*self.nmo, 2*self.nmo))
        Q_full[:self.nmo, :self.nmo] = Qs[0]
        Q_full[self.nmo:, self.nmo:] = Qs[1]
        return Q_full

    def get_preconditioner(self):
        """Compute approximate diagonal of Hessian"""
        # Get Fock matrix in MO basis 
        Fmo_alfa = np.linalg.multi_dot([self.alfa.mo_coeff.T, self.alfa.fock, self.alfa.mo_coeff])
        Fmo_beta = np.linalg.multi_dot([self.beta.mo_coeff.T, self.beta.fock, self.beta.mo_coeff])
        # Set up the Hessian
        no_a = self.alfa.nocc
        nv_a = self.nmo - self.alfa.nocc
        no_b = self.beta.nocc
        nv_b = self.nmo - self.beta.nocc
        hessian = np.zeros((2*self.nmo,2*self.nmo,2*self.nmo,2*self.nmo))
        # Compute the Fock contributions
        for i in range(no_a):
            hessian[no_a:self.nmo,i,no_a:self.nmo,i] += 2 * Fmo_alfa[no_a:,no_a:]
        for a in range(no_a,self.nmo):
            hessian[a,:no_a,a,:no_a] -= 2 * Fmo_alfa[:no_a,:no_a]
        for i in range(self.nmo, no_b):
            hessian[ (self.nmo+no_b):, i, (self.nmo+no_b):,i] += 2 * Fmo_beta[no_b:,no_b:]
        for a in range(self.nmo+no_b,2*self.nmo):
            hessian[ a, self.nmo:(self.nmo+no_b), a, self.nmo:(self.nmo + no_b)] -= 2 * Fmo_beta[:no_b,:no_b]
        # Pick out the diagonal terms
        p, q = np.indices(hessian.shape[:2])
        Q = np.zeros((2*self.nmo, 2*self.nmo))
        Q = hessian[p,q,p,q]
        return np.abs(Q[self.rot_idx]) 

    def diagonalise_fock(self):
        """Diagonalise the Fock matrices via transformation of the generalised eigenvalue problem"""
        # Get the orthogonalisation matrix
        X = self.integrals.orthogonalization_matrix() 
        # Project to linearly independent orbitals
        for spin in [self.alfa, self.beta]:     
            Ft = np.linalg.multi_dot([X.T, spin.fock, X])
            # Diagonalise the Fock matrix
            spin.mo_energy, Ct = np.linalg.eigh(Ft)
            # Transform back to the original basis
            spin.mo_coeff = np.dot(X, Ct)
        # Update densities and Fock matrices
        self.update()

    def transform_vector(self, vec, step, X=None): 
        """ Perform orbital rotation for vector in tangent space""" 
        # Build vector in antisymmetric form 
        kappa = np.zeros((2*self.nmo, 2*self.nmo))
        kappa[self.rot_idx] = vec 
        kappa = kappa - kappa.T  
        if not X is None:   
            kappa = kappa @ X 
            kappa = X.T @ kappa  
        return kappa[self.rot_idx]       

    def try_fock(self, fock_alpha, fock_beta): 
        """Try an extrapolated Fock matrix and update the orbital coefficients"""
        self.alfa.fock = fock_alpha 
        self.beta.fock = fock_beta 
        self.diagonalise_fock()

    def try_fock_vec(self, fock_vec): 
        """Wrapper for try_fock() to handle Fock vectors from DIIS"""
        focks = fock_vec.reshape((2,self.nbsf, self.nbsf))
        alfa_fock = focks[0,:,:].T
        beta_fock = focks[1,:,:].T
        self.try_fock(alfa_fock, beta_fock)

    def get_diis_error(self): 
        """Compute DIIS error vector and DIIS error"""
        vecs=[] 
        err=0 
        for spin in [self.alfa, self.beta]: 
            spin_err_vec = np.linalg.multi_dot([spin.fock, spin.dens, self.integrals.overlap_matrix()])
            spin_err_vec -= spin_err_vec.T
            spin_err_vec = spin_err_vec.T.reshape((-1))
            err+= np.linalg.norm(spin_err_vec)  
            vecs.append(spin_err_vec) 
        err_vec = np.concatenate(vecs)
        return err_vec, err 

    def restore_last_step(self): 
        """Restore orbital coefficients to the previous step"""
        self.alfa.mo_coeff = self.alfa.mo_coeff_save.copy()
        self.beta.mo_coeff = self.beta.mo_coeff_save.copy()
        self.update()
    
    def save_last_step(self): 
        """Save current orbital coefficients"""
        self.alfa.mo_coeff_save = self.alfa.mo_coeff.copy()
        self.beta.mo_coeff_save = self.beta.mo_coeff.copy()
        return 
    
    def take_step(self,step): 
        """Take a step in the orbital spaces"""
        self.save_last_step()
        self.rotate_orb(step) 
    
    def rotate_orb(self,step): 
        """Rotate MO coeffs with a step, step specifies our kpq coefficients"""
        step_a = step[:self.alfa.nrot]
        step_b = step[self.alfa.nrot:(self.alfa.nrot + self.beta.nrot)]
        # Build antisymm step matrices (antihermitian but also real) 
        Ka = np.zeros((self.nmo, self.nmo))
        Kb = np.zeros((self.nmo, self.nmo))
        Ka[self.alfa.rot_idx]  = step_a
        Kb[self.beta.rot_idx]  = step_b
        # Build the Unitary transformations 
        Ua = scipy.linalg.expm(Ka - Ka.T)
        Ub = scipy.linalg.expm(Kb - Kb.T)
        # Transform the coefficients
        self.alfa.mo_coeff = np.dot(self.alfa.mo_coeff, Ua) 
        self.beta.mo_coeff = np.dot(self.beta.mo_coeff, Ub)
        # Update the density and the fock matrices
        self.update()
        
    def uniq_var_indices(self): 
        """ Creates matrices of boolean of size (nmo,nmo)
            Selects nonredudant rotations for the different spin channels """
        mask_alfa = np.zeros((self.nmo, self.nmo), dtype=bool)
        mask_beta = np.zeros((self.nmo, self.nmo), dtype=bool)
        # Includes only occupied-virtual spin-orbital rotations 
        mask_alfa[self.alfa.nocc:,:self.alfa.nocc] = True 
        mask_beta[self.beta.nocc:,:self.beta.nocc] = True 
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
            self.alfa.fock = h1e.copy()
            self.beta.fock = h1e.copy()
        elif(method.lower() == "gwh"):
            # Build GWH guess Hamiltonian
            K = 1.75
            
            self.alfa.fock = np.zeros((self.nbsf,self.nbsf))
            self.beta.fock = np.zeros((self.nbsf,self.nbsf))
            for i in range(self.nbsf):
                for j in range(self.nbsf):
                    self.alfa.fock[i,j] = 0.5 * (h1e[i,i] + h1e[j,j]) * s[i,j]
                    self.beta.fock[i,j] = 0.5 * (h1e[i,i] + h1e[j,j]) * s[i,j]
                    if(i!=j):
                        self.alfa.fock[i,j] *= 1.75
                        self.beta.fock[i,j] *= 1.75
            
        else:
            raise NotImplementedError(f"Orbital guess method {method} not implemented")
        
        # Get orbital coefficients by diagonalising Fock matrix
        self.diagonalise_fock()

        # Compute an asymmetric initial guess
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
              
    def initialise(self, mo_coeff, mat_ci=None, integrals=True):
        """Initialise wavefunction with orbital coefficient (and CI matrix)"""
        pass

    def read_from_disk(self, tag):
        """Read a wavefunction object to disk"""
        pass

    def hamiltonian(self, other):
        """Compute the Hamiltonian coupling with another wavefunction of this type"""
        pass
    
    def approx_hess_on_vec(self, vec, eps=1e-3): 
        """ Compute the approximate Hess * vec product using forward finite difference """
        # Get current gradient
        g0 = self.gradient.copy()
        # Save current position
        mo_save_a, mo_save_b = self.alfa.mo_coeff.copy(), self.beta.mo_coeff.copy()
        # Get forward gradient
        self.take_step(eps * vec)
        g1 = self.gradient.copy()
        # Restore to origin
        self.alfa.mo_coeff, self.beta.mo_coeff = mo_save_a.copy(), mo_save_b.copy()
        self.update()
        # Parallel transport back to current position
        g1 = self.transform_vector(g1, - eps * vec)
        # Get approximation to H @ sk
        return (g1 - g0) / eps

    def mom_update(self, prev_Cs): 
        """ Construct MOM determinant from an old set of orbitals """
        for index, spin in enumerate([self.alfa, self.beta]):
            # Compute projections onto previous occupied space
            prev_Cocc = prev_Cs[index][:,:spin.nocc] 
            p = np.einsum('ij,jk,kl->l', prev_Cocc.T, self.integrals.overlap_matrix(), spin.mo_coeff )
            # Order MOs according to largest projection 
            idx = list(reversed(np.argsort(np.abs(p))))
            spin.mo_coeff = spin.mo_coeff[:,idx]
        
        self.update()

    def mo_cubegen(self, a_idx, b_idx, fname=""): 
        """ Generate and store cube files for specified MOs
                a_idx, b_idx : lists of indices of alpha and beta MOs
        """
        spins = ["a","b"]
        for i, spin in enumerate([self.alfa, self.beta]):    
            for mo in [a_idx,b_idx][i]: 
                cubegen.orbital(self.integrals.mol, fname+f".{spins[i]}.mo.{mo}.cube", spin.mo_coeff[:,mo])

