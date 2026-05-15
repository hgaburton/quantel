from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.uhf import UHF

mol  = PySCFMolecule("mol/formaldehyde.xyz", "6-31g", "angstrom")
ints = PySCFIntegrals(mol,xc="PBE0")
wfn = UHF(ints)
from quantel.opt.lbfgs import LBFGS
wfn.get_orbital_guess(method="gwh")
LBFGS().run(wfn) 
wfn.mo_coeff[0][:,[7,8]] = wfn.mo_coeff[0][:,[8,7]].copy() 
from quantel.opt.gmf import GMF 
GMF().run(wfn, index=1) 
wfn.canonicalize()
wfn.print() 
print("Testing: ", wfn.s2_coupling(wfn)) 
