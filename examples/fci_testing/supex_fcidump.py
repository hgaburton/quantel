import os
import numpy as np
from quantel.ints.fcidump_integrals import FCIDUMP
from quantel.utils.linalg import orthogonalise

# Read FCIDUMP with quantel and check all integrals against PySCF
intdump = FCIDUMP("supex_fcidump")
intdump.print()
print("\n--- Dimensions and electron counts ---")
print("nbsf",  intdump.nbsf())
print("nmo",   intdump.nmo() )
print("nelec", intdump.molecule().nelec())
print("nalfa", intdump.molecule().nalfa())
print("nbeta", intdump.molecule().nbeta())
#print("\n--- Overlap / orthogonalisation matrices ---")
#print("overlap_matrix = I", (intdump.overlap_matrix()))
#print("\n--- Scalar potential ---")
#print("scalar_potential = ", abs(intdump.scalar_potential()))
#print("\n--- One-electron integrals ---")
#oei = intdump.oei_matrix()
#print(oei) 
#print("\n--- Two-electron integrals ---")
#tei = intdump.tei_array()
#print(tei)

from quantel.wfn.cisolver import FCI 
C = np.eye(3) 
mo_integrals = intdump.mo_integrals(C,0,3) 
fci_obj = FCI(mo_integrals,[intdump.molecule().nalfa(),intdump.molecule().nbeta()])
fci_obj.solve(9)
print("Eigvals: ", fci_obj.eigval)
print("Eigvecs: ") 
print(fci_obj.eigvec)
s2s=[]
for i in range(9): 
    s2s.append(float(fci_obj.get_s2(fci_obj.eigvec[:,i].copy())))
print(s2s)
print(fci_obj.get_det_list())
 
