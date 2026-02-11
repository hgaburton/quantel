import numpy as np 
import quantel
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals 
from quantel.utils.linalg import orthogonalise 

mol2 = quantel.Molecule([["H", 0, 0, 0],["H", 0, 0, 1.2]],"angstrom" )
int2 = quantel.LibintInterface("sto-3g",mol2)

mol = PySCFMolecule([["H", 0, 0, 0],["H", 0, 0, 1.2]],"sto-3g", "angstrom" )
ints = PySCFIntegrals(mol) 

#print( ints.overlap_matrix())
Ca = np.array([[1,0],[0,1]], dtype=float)
Cb = np.array([[0,1],[1,0]], dtype=float)
Ca = orthogonalise(Ca, ints.overlap_matrix())
Cb = orthogonalise(Cb, ints.overlap_matrix())
da = np.dot(Ca[:,:1], Ca[:,:1].T)
db = np.dot(Cb[:,:1], Cb[:,:1].T)
print(Ca)
print(Cb)
print(da)
print(db)

vd = np.zeros((2,2,2))
vd[0] = da.copy()
vd[1] = db.copy()


vJ, vK = ints.build_multiple_JK(vd,vd,2,2)
print("Ja,Ka")
print(vJ[0])
print(vK[0])

JKa = ints.build_JK(da)
print("build_JK")
print(JKa)
print(2*vJ[0] - vK[0])
print("LibintJK")
print(int2.build_JK(da))

JKb = ints.build_JK(db)
print("build_JK b")
print(vJ[1])
print(vK[1])
print(JKb)
print(2*vJ[1] - vK[1])
print(int2.build_JK(db))
