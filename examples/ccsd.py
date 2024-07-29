import os
os.environ["OMP_NUM_THREADS"] = "1" 
#os.environ["OPENBLAS_NUM_THREADS"] = "1" 
#os.environ["MKL_NUM_THREADS"] = "1" 
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"
import quantel
import numpy as np
from quantel.wfn.rhf import RHF
from quantel.opt.eigenvector_following import EigenFollow
from quantel.opt.diis import DIIS

np.set_printoptions(linewidth=10000,precision=6,suppress=True)
np.random.seed(7)

class CCSD:
    def __init__(self, ints):
        self.ints = ints
        # Run reference RHF
        self.rhf = RHF(ints)
        self.rhf.get_orbital_guess()
        DIIS().run(self.rhf)

        print()
        print(f"Reference energy = {self.Eref: 10.8f}")

        # Count the orbitals
        self.no = 2 * self.rhf.nocc
        self.nv = 2 * (self.rhf.nmo - self.rhf.nocc)
        self.nb = 2 * self.rhf.nmo

        # Get integrals
        self.get_integrals()

        # Initialise amplitudes
        self.t1 = np.zeros((self.no,self.nv))
        self.t2 = np.zeros((self.no,self.no,self.nv,self.nv))

    @property
    def Eref(self):
        return self.rhf.energy
    
    @property
    def tau(self):
        return 0.5 * self.t2 + np.einsum('ia,jb->ijab',self.t1,self.t1)
    
    @property
    def Ec(self):
        return 0.5 * np.einsum('ijab,ijab',self.tau,self.Voovv)
    
    def get_integrals(self):
        # Orbital coefficients 
        C = self.rhf.mo_coeff.copy()

        # Fock matrix elements
        fmo = np.linalg.multi_dot([C.T,self.rhf.fock,C])
        f = np.zeros((self.nb,self.nb))
        for p in range(self.nb):
            for q in range(self.nb):
                f[p,q] = fmo[p//2,q//2] if (p%2 == q%2) else 0

        # Diagonal terms
        self.eps = np.diag(f)

        # Delta terms
        self.Dov = np.zeros((self.no,self.nv))
        for i in range(self.no):
            for a in range(self.nv):
                self.Dov[i,a] = self.eps[a+self.no] - self.eps[i]
        self.Doovv = np.zeros((self.no,self.no,self.nv,self.nv))
        for i in range(self.no):
            for j in range(self.no):
                for a in range(self.nv):
                    for b in range(self.nv):
                        self.Doovv[i,j,a,b] = self.eps[a+self.no] + self.eps[b+self.no] - self.eps[i] - self.eps[j] 

        # Two-electron integrals
        eri = self.ints.tei_ao_to_mo(C,C,C,C,True,False)
        spin_eri = np.zeros((self.nb,self.nb,self.nb,self.nb))
        for p in range(self.nb):
            for q in range(self.nb):
                for r in range(self.nb):
                    for s in range(self.nb):
                        spin_eri[p,q,r,s] = eri[p//2,q//2,r//2,s//2] if (p%2 == r%2 and q%2 == s%2) else 0
        # Antisymmetrise
        spin_eri = spin_eri - spin_eri.transpose(0,1,3,2)
        # Extract relevant blocks
        self.Voooo = spin_eri[:self.no,:self.no,:self.no,:self.no]
        self.Vooov = spin_eri[:self.no,:self.no,:self.no,self.no:]
        self.Voovv = spin_eri[:self.no,:self.no,self.no:,self.no:]
        self.Vovvo = spin_eri[:self.no,self.no:,self.no:,:self.no]
        self.Vovvv = spin_eri[:self.no,self.no:,self.no:,self.no:]
        self.Vvvvv = spin_eri[self.no:,self.no:,self.no:,self.no:]

    def get_intermediates(self):
        # h intermediates for CCSD
        # Taken from Scuseria and Schaefer, J. Chem. Phys. 90, 3700 (1989)
        tau = self.tau
        # Equation (5)
        self.hvv = np.diag(self.eps[self.no:]) - np.einsum('jkbc,jkac->ba',self.Voovv,tau)
        # Equation (6)
        self.hoo = np.diag(self.eps[:self.no]) + np.einsum('jkbc,ikbc->ij',self.Voovv,tau)
        # Equation (7) 
        self.hvo = np.einsum('jkbc,kc->bj',self.Voovv,self.t1)
        # Equation (9)
        self.gvv = self.hvv + np.einsum('kadc,kd->ca',self.Vovvv,self.t1)
        # Equation (10)
        self.goo = self.hoo + np.einsum('klic,lc->ik',self.Vooov,self.t1)
        # Equation (11)
        self.aoooo = (self.Voooo + np.einsum('klic,jc->ijkl',self.Vooov,self.t1) 
                                 - np.einsum('kljc,ic->ijkl',self.Vooov,self.t1)
                                 + np.einsum('klcd,ijcd->ijkl',self.Voovv,tau))
        # Equation (12)
        self.bvvvv = (self.Vvvvv - np.einsum('kadc,kb->abcd',self.Vovvv,self.t1)
                                 + np.einsum('kbdc,ka->abcd',self.Vovvv,self.t1))
        # Equation (13)
        self.hovvo = (self.Vovvo - np.einsum('lkic,la->icak',self.Vooov,self.t1)
                                 + np.einsum('kacd,id->icak',self.Vovvv,self.t1)
                                 - np.einsum('klcd,ilda->icak',self.Voovv,tau))


    def r1_residual(self):
        # Equation 4 in Scuseria and Schaefer, J. Chem. Phys. 90, 3700 (1989)
        # r1 = h_b^a * t_i^b - h_i^j * t_j^a + h_b^j * (t_{ij}^{ab} + t_i^b * t_j^a)
        #    + <ib||aj> t_j^b - <ja||bc> tau_{ij}^{bc} - <jk||ib> tau_{jk}^{ab}
        tau = self.tau
        teff = self.t2 + np.einsum('ib,ja->ijab',self.t1,self.t1)
        return (np.einsum('ba,ib->ia',self.hvv,self.t1)
              - np.einsum('ij,ja->ia',self.hoo,self.t1)
              + np.einsum('bj,ijab->ia',self.hvo,teff)
              + np.einsum('ibaj,jb->ia',self.Vovvo,self.t1)
              - np.einsum('jabc,ijbc->ia',self.Vovvv,tau)
              - np.einsum('jkib,jkab->ia',self.Vooov,tau))
    
    def r2_residual(self):
        # Equation 8 in Scuseria and Schaefer, J. Chem. Phys. 90, 3700 (1989)
        # r2 = <ij||ab> + a_{ij}^{kl} tau_{kl}^{ab} + b_{cd}^{ab} tau_{ij}^{cd}
        #    + P(ab) (g_c^a t_{ij}^{cb} + <ka||ij> t_j^b)
        #    - P(ij) (g_i^k t_{kj}^{ab} - <ab||ci> t_i^c)
        #    + P(ij) P(ab) (h_{ic}^{ak} t_{jk}^{bc} - <ic||ak> t_j^c t_k^b)
        tau = self.tau
        int1 = np.einsum('ca,ijcb->ijab',self.gvv,self.t2) + np.einsum('ijka,kb->ijab',self.Vooov,self.t1)
        int2 = np.einsum('ik,kjab->ijab',self.goo,self.t2) - np.einsum('jcba,ic->ijab',self.Vovvv,self.t1)
        int3 = np.einsum('icak,jkbc->ijab',self.hovvo,self.t2) - np.einsum('icak,jc,kb->ijab',self.Vovvo,self.t1,self.t1)
        return (self.Voovv + np.einsum('ijkl,klab->ijab',self.aoooo,tau)      
                           + np.einsum('cdab,ijcd->ijab',self.bvvvv,tau)
                           + int1 - int1.transpose(0,1,3,2)
                           - int2 + int2.transpose(1,0,2,3)
                           + int3 - int3.transpose(0,1,3,2) 
                           - int3.transpose(1,0,2,3) + int3.transpose(1,0,3,2))   

    def run(self):
        # Reset amplitudes
        self.t1 *= 0
        self.t2 *= 0

        # Get MP1 amplitude guess
        self.t2 = - self.Voovv / self.Doovv

        # Print reference energy
        print()
        print(f"MP2 correlation energy (Eh) = {self.Ec: 10.8f}")
        print(f"      MP2 total energy (Eh) = {self.Eref+self.Ec: 10.8f}")
        print()

        # Set the tolerance
        tol = 1.0e-6
        kappa = 0.8

        # Perform CC iterations
        for i in range(100):
            # Compute the intermediates
            self.get_intermediates()

            # Compute 1-body resisual
            r1 = self.r1_residual()
            r2 = self.r2_residual()

            # Compute error
            err = np.linalg.norm(r1) + np.linalg.norm(r2)

            # Print current energy and error
            print(f" {i+1:5d}   {self.Eref+self.Ec: 12.8f}    {err: 12.3e}")
            if(err < tol):
                break

            # Compute step (no diis)
            dt1 = - r1 / self.Dov
            dt2 = - r2 / self.Doovv
             
            alpha = 0.5
            p = 2
            dt1 = - r1 / (self.Dov + np.power(alpha + self.Dov, -p))
            dt2 = - r2 / (self.Doovv + np.power(alpha + self.Doovv, -p))
            #dt2 = - r2 * np.power(1 - np.exp(-kappa * self.Doovv), 2) / self.Doovv

            # Increment amplitudes
            self.t1 += dt1
            self.t2 += dt2

if __name__ == "__main__":
    # Initialise molecular structure (linear H4)
    R = 4.0
    mol = quantel.Molecule([["H",0.0,0.0,0.0],
                            ["H",0.0,0.0,R],
                            ["H",0.0,0.0,2*R],
                            ["H",0.0,0.0,3*R]], 'angstrom')
    mol.print()

    # Initialise interface to Libint2
    libints = quantel.LibintInterface("6-31g",mol)

    # Build CCSD object
    ccsd = CCSD(libints)
    ccsd.run()