import quantel
import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.rhf import RHF
from quantel.wfn.uhf import UHF
import matplotlib.pyplot as plt 

rhf_energies=[]
uhf_energies=[]

bond_lens = np.linspace(0.3,6,60) 
for bond_len in bond_lens:
    mol  = PySCFMolecule([["H", 0, 0, 0],["H", 0, 0, bond_len]], "6-31g", "angstrom")
    ints = PySCFIntegrals(mol) # so here we dont put an exchange correlation functional in so it doesnt matter...
    rhf_wfn = RHF(ints)
    uhf_wfn = UHF(ints)
    # Setup optimiser
    from quantel.opt.lbfgs import LBFGS
    from quantel.opt.diis import DIIS
    rhf_wfn.get_orbital_guess(method="gwh")
    uhf_wfn.get_orbital_guess(method="core", asymmetric=True)
    LBFGS().run(rhf_wfn)
    LBFGS().run(uhf_wfn)
    #DIIS().run(rhf_wfn)
    #DIIS().run(uhf_wfn)
    # Test canonicalisation and Hessian eigenvalue
    rhf_wfn.canonicalize()
    rhf_energies.append(rhf_wfn.energy)
    uhf_wfn.canonicalize()
    uhf_energies.append(uhf_wfn.energy)
    # Test Hessian index
    #wfn.get_davidson_hessian_index()

fig, ax = plt.subplots()
ax.plot(bond_lens, rhf_energies, label="RHF") 
ax.plot(bond_lens, uhf_energies, label="UHF")
plt.legend()
plt.savefig("./H2_energies.png")  
 
