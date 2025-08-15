import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.rhf import RHF
from quantel.opt.lbfgs import LBFGS
from tups import T_UPS
from quantel.opt.linear import Linear


mol  = PySCFMolecule("h6.xyz", "sto-3g", "angstrom")
ints = PySCFIntegrals(mol)

wfn = RHF(ints)
wfn.get_orbital_guess(method="gwh")

np.set_printoptions(linewidth=1000,suppress=True,precision=6)
LBFGS().run(wfn)

np.random.seed(10)
trials = 15
data = np.zeros((1,2))
opt = LBFGS(with_transport=False,with_canonical=False,prec_thresh=0.1)
lin = Linear()
for isample in range(trials):
    tUPS = T_UPS(wfn, include_doubles=True, approx_prec=True, use_prec=True, pp=False, oo=True, layers=3)
    tUPS.get_initial_guess()
    print(f"Use preconditioner: {tUPS.use_prec}")
    print(f"Approximate preconditioner: {tUPS.approx_prec}")
    print(f"Orbital Optimised: {tUPS.orb_opt}")
    print(f"Perfect Pairing: {tUPS.perf_pair}")
    iterations, energy = opt.run(tUPS, maxit=2000)
    data[0,:] = iterations, energy
    with open("../dump/h6_mol/H6_2.00/random/linear-dogleg.csv", "ab") as f:
            np.savetxt(f, data, delimiter=",")

# opt.run(tUPS, maxit=2000)

# iterations, energy = lin.run_linesearch(tUPS, maxit=2000)

# energies, eigenvectors = np.linalg.eigh(tUPS.mat_H.todense())
# print(f"exact = {energies}")

quit()
# In the MO basis
# hmo = ints.oei_ao_to_mo(Cmo,Cmo)
# print(hmo)
# eri = ints.tei_ao_to_mo(Cmo,Cmo,Cmo,Cmo,True,False)
# print(eri.shape)
# print(eri[0,1,0,1], eri[1,0,1,0])
# print(eri[1,3,2,4], eri[3,1,4,2])

F = hmo + 2*np.einsum('piqi->pq',eri[:,:3,:,:3]) - np.einsum('piiq->pq',eri[:,:3,:3,:])
print(F)
P = np.diag([1,1,1,0,0,0])

print(np.trace((F+hmo) @ P) + ints.scalar_potential())

# To build your own Hamiltonian
vnuc = ints.scalar_potential()
hpq  = ints.oei_ao_to_mo(Cmo,Cmo) # <p|h|q>
eri_pqrs = ints.tei_ao_to_mo(Cmo,Cmo,Cmo,Cmo,True,False) # <pq|rs>
print(hpq)
# Make sure you include two-electron contributions correctly for different spins
# <aa|aa>, <ab|ab>, <ba|ba>, <bb|bb>