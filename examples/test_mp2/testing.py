from quantel.ints.pyscf_integrals import PySCF_MO_Integrals, PySCFIntegrals, PySCFMolecule
from quantel.wfn.csf import CSF
from quantel.wfn.cisolver import FCI
from quantel.wfn.uhf import UHF
from quantel.opt.lbfgs import LBFGS
from quantel.utils.linalg import matrix_print 
import numpy as np 
# Setup molecule
mol = PySCFMolecule("../mol/formaldehyde.xyz", "sto-3g", "angstrom",spin=0,charge=0)
ints = PySCFIntegrals(mol)

# Run RHF to get MO coefficients
wfn = CSF(ints,"+-")
wfn.get_orbital_guess(method="gwh")
LBFGS().run(wfn)
wfn.canonicalize()
wfn.get_fock() 
fock_ao = wfn.gen_coupling 
fock_mo = np.linalg.multi_dot((wfn.mo_coeff.T, fock_ao, wfn.mo_coeff )) 
matrix_print(fock_mo)

 
# check against PYSCF
