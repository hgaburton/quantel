from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.csf import CSF
from quantel.utils.linalg import matrix_print
from quantel.gnme.csf_noci import csf_rdm1 
import numpy as np 

mol  = PySCFMolecule("mol/formaldehyde.xyz", "def2svp", "angstrom")
ints = PySCFIntegrals(mol)

# Initialise CSF object for an open-shell singlet state
wfn = CSF(ints, '+-')
wfn.get_orbital_guess(method="gwh")
from quantel.opt.lbfgs import LBFGS
wfn.get_orbital_guess()
LBFGS().run(wfn)
wfn.canonicalize()
rdm1 = wfn.vd[0] + wfn.vd[1] + wfn.vd[2] 
 
s, other_rdm1 = csf_rdm1(wfn, wfn, ints.overlap_matrix(), thresh=1e-10)
print(rdm1)
print(np.allclose(rdm1, wfn.mo_coeff @ other_rdm1 @ wfn.mo_coeff.T )) 
