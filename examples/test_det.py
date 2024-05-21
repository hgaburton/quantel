import quantel
import numpy as np
from quantel.wfn.rhf import RHF
from quantel.opt.eigenvector_following import EigenFollow

np.set_printoptions(linewidth=10000,precision=6,suppress=True)
np.random.seed(7)

# Initialise molecular structure (square H4)
mol = quantel.Molecule([["H",0.0,0.0,0.0],
                        ["H",0.0,0.0,1.0],
                        ["H",0.0,0.0,2.0],
                        ["H",0.0,0.0,3.0]])
print(mol.natom())
mol.print()

# Initialise interface to Libint2
ints = quantel.LibintInterface("6-31g",mol)

# Initialise RHF object from integrals
wfn = RHF(ints)
wfn.get_orbital_guess()

# Run eigenvector-following to target a minimum
EigenFollow().run(wfn, index=0)
# Print some information about optimal solution
print(f"\nEnergy = {wfn.energy: 16.10f}")
print("\nMO coefficients:")
print(wfn.mo_coeff)

# Get number of MOs and coefficients
nmo = ints.nmo()
wfn.canonicalize()
C = wfn.mo_coeff.copy()
no=2

# Construct MO integral objectPerform MO transfrom
mo_ints = quantel.MOintegrals(C,C,ints)
#print(mo_ints.oei_matrix(True))

# Check the mo_ints are all good
print("mo_ints.oei_matrix(True)")
h1a = mo_ints.oei_matrix(True)
print(h1a)
h1b = mo_ints.oei_matrix(False)
print(h1b)
h2aa = mo_ints.tei_array(True,True)
h2ab = mo_ints.tei_array(True,False)
h2bb = mo_ints.tei_array(False,False)

en = ints.scalar_potential()
en += np.einsum('pp',h1a[:no,:no])
en += np.einsum('pp',h1b[:no,:no])
print(en)
en += 0.5 * np.einsum('pqpq',h2aa[:no,:no,:no,:no])
en += 0.5 * np.einsum('pqpq',h2ab[:no,:no,:no,:no])
en += 0.5 * np.einsum('qpqp',h2ab[:no,:no,:no,:no])
en += 0.5 * np.einsum('pqpq',h2bb[:no,:no,:no,:no])

# Build CI object
cispace = quantel.CIspace(nmo,2,2,'FCI')
cimanip = quantel.CIexpansion(mo_ints,cispace)
vec = np.zeros(cispace.ndet())
vec[0] = 1.0
cispace.print_vector(vec,1e-10)
sig = cimanip.sigma_vector(vec)
print(sig)
cispace.print_vector(sig,1e-10)
print("Sig",np.dot(sig, vec))
print("Npy",en)
quit()

# Define the CI space
detlist = []
detlist.append(quantel.Determinant([1,1,0,0],[1,1,0,0]))
#detlist.append(quantel.Determinant([1,1,0,0],[1,0,1,0]))
#detlist.append(quantel.Determinant([1,0,1,0],[1,1,0,0]))
print(detlist[0])
print(detlist[0].bitstring())
ci.define_space(detlist)
ci.print()

v  = np.array([1.])
v /= np.linalg.norm(v)
v = v.tolist()

# Print CI vector
ci.print_vector(v)

# Get sigma vector
sig = ci.sigma_vector(v)
print(sig)
ci.print_vector(sig)
print("Energy: ", en)
print("Sigma : ",np.dot(sig, v) )
quit()


detlist = []
detlist.append(quantel.Determinant([1,1,1,0,0,0,0],[1,1,0,1,0,0,0]))
detlist.append(quantel.Determinant([1,1,1,0,0,0,0],[1,1,1,0,0,0,0]))
detlist.append(quantel.Determinant([1,1,0,1,0,0,0],[1,1,1,0,0,0,0]))
coeff = [1.,0.,0.]

ci = quantel.CIexpansion(detlist, coeff)
print("PRINT")
ci.print(-1)
