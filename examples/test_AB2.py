from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.rhf import RHF
from quantel.opt.lbfgs import LBFGS

print("\n===============================================")
print(f" Testing AB2 Orbital construction on RHF object")
print("===============================================")
# Setup molecule and integrals 
mol  = PySCFMolecule("./mol/formaldehyde.xyz", "6-31g", "angstrom")
ints = PySCFIntegrals(mol)

# Optimise onto RHF ground state 
wfn = RHF(ints)
wfn.get_orbital_guess(method="gwh")
LBFGS().run(wfn)
ref_mo = wfn.mo_coeff.copy() 

# Testing single and smushed AB2 construction 
for i, indices in enumerate([ None, [0,1,2,3]]):
    AB2_orbitals = wfn.compute_and_update_AB2(indices=indices)
    #wfn.mo_cubegen([8,9,10,11],f"{i}.AB2")
