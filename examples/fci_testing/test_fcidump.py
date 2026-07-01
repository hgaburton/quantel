import os
import numpy as np
from quantel.ints.fcidump_integrals import FCIDUMP
from quantel.utils.linalg import orthogonalise
# Read FCIDUMP with quantel and check all integrals against PySCF
intdump = FCIDUMP("twosite_hubb_fcidump")
intdump.print()
print("\n--- Dimensions and electron counts ---")
print("nbsf",  intdump.nbsf())
print("nmo",   intdump.nmo() )
print("nelec", intdump.molecule().nelec())
print("nalfa", intdump.molecule().nalfa())
print("nbeta", intdump.molecule().nbeta())
print("\n--- Overlap / orthogonalisation matrices ---")
print("overlap_matrix = I", (intdump.overlap_matrix()))
print("\n--- Scalar potential ---")
print("scalar_potential = ", abs(intdump.scalar_potential()))
print("\n--- One-electron integrals ---")
oei = intdump.oei_matrix()
print(oei) 
print("\n--- Two-electron integrals ---")
tei = intdump.tei_array()
print(tei)

from quantel.wfn.cisolver import FCI 
C = np.eye(2) 
mo_integrals = intdump.mo_integrals(C,0,2) 
fci_obj = FCI(mo_integrals,[intdump.molecule().nalfa(),intdump.molecule().nbeta()])
fci_obj.solve(4)
print("Eigvals: ", fci_obj.eigval)
print("Eigvecs: ") 
print(fci_obj.eigvec)
s2s=[]
for i in range(4):
    vec = fci_obj.eigvec[:,i].copy()  
    print("=======")
    print("solution ",i)
    print(vec)
    s2, rdm2ab = fci_obj.get_s2(vec)
    print(s2) 
print(fci_obj.get_det_list())
test = np.zeros(4,dtype=float)
test[1] = -1/np.sqrt(2) 
test[2] = 1/np.sqrt(2)
s2, array = fci_obj.get_s2(test) 
print("=======")
print("Difference: ", np.abs(test - fci_obj.eigvec[:,1]))
print("Test vec:", test) 
print("Test S2: ", s2)  
