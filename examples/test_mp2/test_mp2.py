from quantel.ints.pyscf_integrals import PySCF_MO_Integrals, PySCFIntegrals, PySCFMolecule
from quantel.wfn.csf import CSF
from quantel.wfn.cisolver import FCI
from quantel.opt.lbfgs import LBFGS
from quantel.utils.linalg import matrix_print 
import numpy as np 

# Setup molecule
mol = PySCFMolecule("../mol/bh2.xyz", "sto-3g", "angstrom",spin=1,charge=0)
ints = PySCFIntegrals(mol)

# Run RHF to get MO coefficients
wfn = CSF(ints,"+")
wfn.get_orbital_guess(method="gwh")
LBFGS().run(wfn)

# Build the integrals
mo_ints = PySCF_MO_Integrals(ints)
mo_ints.update_orbitals(wfn.mo_coeff,0,ints.nmo())

# Setup and solve FCI
ci = FCI(mo_ints, (mol.nalfa(), mol.nbeta()), version=1)
# Only create excitations with same ms value
#print(ci.ndet) 
#print list of all the determinants
#ci.cispace.print() 
Hamiltonian = ci.get_hamiltonian() 

#matrix_print(Hamiltonian[:,:2])
# This contains the Hamiltonian coupling matrix elements but to calc the MP2 amplitude we also need the fock matrix elements 
# since the orbtials are eigenfunctions of the fock operator the determinants are eigenfunctions with eigenvalues as the sum of single orbital energies

# First order correction to the wave function: 
wfn1 = np.zeros((ci.ndet), dtype=float) 
wfn1[0]=1 
for i in range(1, ci.ndet): 
   wfn1[i] = Hamiltonian[i,0]/(Hamiltonian[0,0] - Hamiltonian[i,i])   
wfn1 /= np.linalg.norm(wfn1) 

# Second order energy corrections 
# First order energy correction is just <psi| \hat(V) |psi>, so E = E(0) + E(1)
E2 = 0.0 
# only need to look through the double excitations since the Hamiltonian needs to connect to this. 
for i in range(1, ci.ndet): 
   E2 += (np.abs(Hamiltonian[i,0])**2)/(Hamiltonian[0,0] - Hamiltonian[i,i])   

print("Zeroth order energy:",Hamiltonian[0,0])
print("Correlation energy:",E2)
print("MP2 Energy:", E2+ wfn.energy) 
