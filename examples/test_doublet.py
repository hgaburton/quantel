import quantel
import numpy as np
from quantel.wfn.rhf import RHF
from quantel.wfn.cisolver import FCI, CIS, CustomCI
from quantel.opt.lbfgs import LBFGS
import datetime

np.set_printoptions(linewidth=10000,precision=6,suppress=True,edgeitems=5)
np.random.seed(7)

# Initialise molecular structure (square H4)
R = 3
mol = quantel.Molecule([["H",1.0*R,0.0,0.0],
                        ["H",0.0,1.0*R,0.0],
                        ["H",0.0,0.0,1.0*R]
                       ],"bohr")
print("Molecule:")
mol.print()

# Initialise interface to Libint2
ints = quantel.LibintInterface("6-31g", mol)

# Initialise RHF object from integrals
wfn = RHF(ints)
wfn.get_orbital_guess(method='gwh')

# Get orbital coefficients
C = wfn.mo_coeff.copy()
print("MO coefficients")
print(C)

# Setup MO ints
oei = ints.oei_ao_to_mo(C,C,True)
tei = ints.tei_ao_to_mo(C,C,C,C,True,False)
mo_ints = quantel.MOintegrals(ints.scalar_potential(),oei,tei,ints.nmo())
nelec = (mol.nalfa()+2, mol.nbeta()+2)
nelec = (5,4)

print("<pi|qi>")
print(tei[:,0,:,0])
print("<pi|iq>")
print(tei[:,0,0,:])

fci1 = CustomCI(mo_ints, ["2222a0","22220a"], nelec)
fci1.cispace.print()
e,x = fci1.solve(nroots=2)

th = 0.4
x = np.array([np.cos(th), np.sin(th)])
x = np.array([1,-1])
x = x / np.linalg.norm(x)


fci1.cispace.print_vector(x,1e-6)
rdm1a = fci1.cispace.rdm1(x,True)
rdm1b = fci1.cispace.rdm1(x,False)
rdm1 = rdm1a + rdm1b
print(rdm1)
rdm2aa = fci1.cispace.rdm2(x,True,True)
rdm2ab = fci1.cispace.rdm2(x,True,False)
rdm2ba = rdm2ab.transpose(1,0,3,2)
rdm2bb = fci1.cispace.rdm2(x,False,False)
rdm2 = rdm2aa + rdm2ab + rdm2ba + rdm2bb 
for p in range(ints.nmo()):
    for q in range(ints.nmo()):
        for r in range(ints.nmo()):
            for s in range(ints.nmo()):
                if(abs(rdm2[p,q,r,s]) > 1e-6): 
                    print(f"rdm2[{p},{q},{r},{s}] = {rdm2[p,q,r,s]: 8.1f}   eri[{p},{q},{r},{s}] = {tei[p,q,r,s]: 16.10f}")

print(oei)


# Now make an approximation
nmo = ints.nmo()
rdm1new = np.diag(np.diag(rdm1))
rdm2new = np.zeros((nmo,nmo,nmo,nmo))
for p in range(nmo):
    for q in range(nmo):
        rdm2new[p,q,p,q] = rdm2[p,q,p,q]
        rdm2new[p,q,q,p] = rdm2[p,q,q,p] 
a = np.einsum('pqpq->pq',rdm2new)
b = np.einsum('pqqp->pq',rdm2new)
print(rdm1new)
print(a)
print(b)

print("Full generalized Fock matrix")
F1 = np.einsum('pr,rq->pq',rdm1,oei) + np.einsum('prst,stqr->pq',rdm2,tei)
print(F1)
print(2 * (F1 - F1.T))
print("Alt generalized Fock matrix")
F2 = np.einsum('pr,rq->pq',rdm1new,oei) + np.einsum('prst,stqr->pq',rdm2new,tei)
print(F2)
print(2 * (F2 - F2.T))

# Build ensemble densities
rdm1 = np.zeros((nmo,nmo))
rdm2 = np.zeros((nmo,nmo,nmo,nmo))
for x in np.eye(fci1.cispace.ndet()):
    print(x)
    rdm1a = fci1.cispace.rdm1(x,True)
    rdm1b = fci1.cispace.rdm1(x,False)
    rdm2aa = fci1.cispace.rdm2(x,True,True)
    rdm2ab = fci1.cispace.rdm2(x,True,False)
    rdm2ba = rdm2ab.transpose(1,0,3,2)
    rdm2bb = fci1.cispace.rdm2(x,False,False)
    rdm1 += (rdm1a + rdm1b) / fci1.cispace.ndet()
    rdm2 += (rdm2aa + rdm2ab + rdm2ba + rdm2bb) / fci1.cispace.ndet()

print("Ensemble generalized Fock matrix")
F1 = np.einsum('pr,rq->pq',rdm1,oei) + np.einsum('prst,stqr->pq',rdm2,tei)
print(F1)
print(2 * (F1 - F1.T))
quit()
# Find the RHF minimum
LBFGS().run(wfn)

# Build the integrals
C = wfn.mo_coeff.copy()
oei = ints.oei_ao_to_mo(C,C,True)
tei = ints.tei_ao_to_mo(C,C,C,C,True,False)

# Setup the MO integral space
mo_ints = quantel.MOintegrals(ints.scalar_potential(),oei,tei,ints.nmo())
nelec = (mol.nalfa(), mol.nbeta())
print(nelec)

tstart = datetime.datetime.now()
print(" Building CI space and Hamiltonian...")
fci1 = CIS(mo_ints, nelec, version=1)
tmid = datetime.datetime.now()
print("Time to initialize FCI (version 1): ", (tmid - tstart).total_seconds())
fci2 = CIS(mo_ints, nelec, version=2)
tend = datetime.datetime.now()
print("Time to initialize FCI (version 2): ", (tend - tmid).total_seconds())

vtest = np.random.rand(fci1.ndet)
tstart = datetime.datetime.now()
ham1 = fci1.cispace.build_Hmat()
tmid = datetime.datetime.now()
print("Time to build H (version 1): ", (tmid - tstart).total_seconds())
ham2 = fci2.cispace.build_Hmat()
tend = datetime.datetime.now()
print("Time to build H (version 2): ", (tend - tmid).total_seconds())
print("Hamiltonian difference norm: ", np.linalg.norm(ham1 - ham2))
print(ham1)
print(ham2)
# Show where they differ
diff = np.abs(ham1 - ham2)
for i in range(ham1.shape[0]):
    for j in range(ham1.shape[1]):
        if(diff[i,j] > 1e-8):
            print(f"Difference at ({i},{j}): {ham1[i,j]} vs {ham2[i,j]}")

tstart = datetime.datetime.now()
hv1 = fci1.cispace.H_on_vec(vtest)
tend = datetime.datetime.now()
print("Time to apply H (version 1): ", (tend - tstart).total_seconds())
hv2 = fci2.cispace.H_on_vec(vtest)
tend2 = datetime.datetime.now()
print("Time to apply H (version 2): ", (tend2 - tend).total_seconds())
print("H application difference norm: ", np.linalg.norm(hv1 - hv2))
print(hv1)
print(hv2)
print(ham1 @ vtest)


# test build of Hd
tstart = datetime.datetime.now()
Hd1 = fci1.cispace.build_Hd()
tmid = datetime.datetime.now()
print("Time to build Hd (version 1): ", (tmid - tstart).total_seconds())
Hd2 = fci2.cispace.build_Hd()
tend = datetime.datetime.now()
print("Time to build Hd (version 2): ", (tend - tmid).total_seconds())
print("Hd difference norm: ", np.linalg.norm(Hd1 - Hd2))
print(Hd1)
print(Hd2)
print(np.diag(ham2))

# Solve first problem
nroots = 5
e,x = fci1.solve(nroots,verbose=5)
print(e)
x1 = np.copy(x[:,3])
print(x1)
# Solve second problem
e,x = fci2.solve(nroots,verbose=5)
print(e)
x2 = np.copy(x[:,3])
print(x2)
print("Exact energies")
print(np.linalg.eigh(ham2)[0])
# Compare the results
tstart = datetime.datetime.now()
rdm1 = fci1.cispace.rdm2(x1,True,True)
#rdm1 = fci1.cispace.rdm1(x1,True)
tend = datetime.datetime.now()
print("Time to build RDM (version 1): ", (tend - tstart).total_seconds())
rdm2 = fci2.cispace.rdm2(x2,True,True)
#rdm2 = fci2.cispace.rdm1(x2,True)
tend2 = datetime.datetime.now()
print("Time to build RDM (version 2): ", (tend2 - tstart).total_seconds())
print(" RDM difference norm: ", np.linalg.norm(rdm1 - rdm2))
# show differences
for i in range(rdm1.shape[0]):
    for j in range(rdm1.shape[1]):
        for k in range(rdm1.shape[2]):
            for l in range(rdm1.shape[3]):
                if(np.abs(rdm1[i,j,k,l] - rdm2[i,j,k,l]) > 1e-8):
                    print(f"Difference at ({i},{j},{k},{l}): {rdm1[i,j,k,l]} vs {rdm2[i,j,k,l]}")
quit()
