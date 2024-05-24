import quantel
import numpy as np
from quantel.wfn.rhf import RHF
from quantel.opt.eigenvector_following import EigenFollow

np.set_printoptions(linewidth=10000,precision=6,suppress=True)
np.random.seed(7)

# Initialise molecular structure (square H4)
R = 2.0 * 1.8897259886
mol = quantel.Molecule([["H",0.0,0.0,0.0*R],
                        ["H",0.0,0.0,1.0*R],
                        ["H",0.0,0.0,2.0*R],
                        ["H",0.0,0.0,3.0*R],
                        ["H",0.0,0.0,4.0*R],
                        ["H",0.0,0.0,5.0*R],
                        ["H",0.0,0.0,6.0*R],
                        ["H",0.0,0.0,7.0*R]])
print(mol.natom())
no = 4
ncore = 1
nact = 6
ne = no - ncore
mol.print()

# Initialise interface to Libint2
ints = quantel.LibintInterface("sto-3g",mol)

# Initialise RHF object from integrals
wfn = RHF(ints)
wfn.get_orbital_guess()

# Run eigenvector-following to target a minimum
EigenFollow().run(wfn, index=0)

# Get number of MOs and coefficients
nmo = ints.nmo()
wfn.canonicalize()
C = wfn.mo_coeff.copy()
# Print some information about optimal solution
print(f"\nNuclear repulsion = {ints.scalar_potential(): 16.10f}")
print(f"Energy = {wfn.energy: 16.10f}")
print("\nMO coefficients:")
print(wfn.mo_coeff)
Fao = wfn.fock
Fmo = np.linalg.multi_dot([C.T, Fao, C])

# Construct MO integral objectPerform MO transfrom
print("Initialise MO integrals")
mo_ints = quantel.MOintegrals(ints)
print("Compute integrals")
mo_ints.update_orbitals(C,ncore,nact)
print("Done")

#print(mo_ints.oei_matrix(True))

# Check the mo_ints are all good
h1a = mo_ints.oei_matrix(True)
h1b = mo_ints.oei_matrix(False)
h2aa = mo_ints.tei_array(True,True)
h2ab = mo_ints.tei_array(True,False)
h2bb = mo_ints.tei_array(False,False)

print(h1a.shape)
print(h2aa.shape)

en = mo_ints.scalar_potential()
en += np.einsum('pp',h1a[:ne,:ne])
en += np.einsum('pp',h1b[:ne,:ne])
en += 0.5 * np.einsum('pqpq',h2aa[:ne,:ne,:ne,:ne])
en += 0.5 * np.einsum('pqpq',h2ab[:ne,:ne,:ne,:ne])
en += 0.5 * np.einsum('qpqp',h2ab[:ne,:ne,:ne,:ne])
en += 0.5 * np.einsum('pqpq',h2bb[:ne,:ne,:ne,:ne])

# Build CI object
print("Initialise CI object... ")
cispace = quantel.CIspace(mo_ints,no-ncore,no-ncore,'FCI')
print("done")

vec = np.zeros(cispace.ndet())
vec[0] = 1.0
#print("\nCI vector")
#cispace.print_vector(vec,1e-10)
sig = cispace.H_on_vec(vec)
#print("\nSigma vector")
#cispace.print_vector(sig,1e-10)

print(f"\nEnergy from np.einsum    = {en: 16.10f}")
print(f"Energy from sigma vector = {np.dot(sig, vec): 16.10f}")

print("Build Hmat from sigma vector")
#ndet = cispace.ndet()
#Hmat = np.zeros((ndet,ndet))
#for irow in range(ndet):
#    print(irow, ndet)
#    vec = np.zeros(cispace.ndet())
#    vec[irow] = 1.0
#    Hmat[irow] = cispace.H_on_vec(vec).copy()
print("Sigma Hmat")
#print(Hmat[:21,:21])

print("Building CI space Hmat")
Hfci = cispace.build_Hmat()
print("Build_Hmat")
print(Hfci[:21,:21])
print("Solve eigenvalue problem")
eci, vci = np.linalg.eigh(Hfci)
print(eci[:10])

print("Build density matrices")
vgs = vci[:,0].copy()
rdm1_a = cispace.rdm1(vgs,True)
rdm1_b = cispace.rdm1(vgs,False)
rdm1 = rdm1_a + rdm1_b
rdm2_aa = cispace.rdm2(vgs,True,True)
rdm2_ab = cispace.rdm2(vgs,True,False)
rdm2_bb = cispace.rdm2(vgs,False,False)
print(rdm1)

en = mo_ints.scalar_potential()
en += np.einsum('pq,pq',rdm1_a,h1a)
en += np.einsum('pq,pq',rdm1_b,h1b)
en += 0.25 * np.einsum('pqrs,pqrs',h2aa,rdm2_aa)
en += 1.00 * np.einsum('pqrs,pqrs',h2ab,rdm2_ab)
en += 0.25 * np.einsum('pqrs,pqrs',h2bb,rdm2_bb)

print(f"RDM energy = {en: 16.10f}")

v1 = np.random.rand(cispace.ndet())
v2 = np.random.rand(cispace.ndet())
v1 /= np.linalg.norm(v1)
v2 /= np.linalg.norm(v2)

trdm1_a = cispace.trdm1(v1,v2,True)
trdm1_b = cispace.trdm1(v1,v2,False)
trdm2_aa = cispace.trdm2(v1,v2,True,True)
trdm2_ab = cispace.trdm2(v1,v2,True,False)
trdm2_bb = cispace.trdm2(v1,v2,False,False)

H12 = v1.dot(v2) * mo_ints.scalar_potential()
H12 += np.einsum('pq,pq',trdm1_a,h1a)
H12 += np.einsum('pq,pq',trdm1_b,h1b)
H12 += 0.25 * np.einsum('pqrs,pqrs',h2aa,trdm2_aa)
H12 += 1.00 * np.einsum('pqrs,pqrs',h2ab,trdm2_ab)
H12 += 0.25 * np.einsum('pqrs,pqrs',h2bb,trdm2_bb)
print(H12)
print(np.linalg.multi_dot([v1.T, Hfci, v2]))


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
