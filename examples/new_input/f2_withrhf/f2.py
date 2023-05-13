import numpy as np
from pyscf import gto, scf

mol = gto.M(
    atom=f"F 0 0 0; F 0 0 1.7",
    basis='6-31g',
    symmetry=True,
    unit="Angstrom")

mf = scf.RHF(mol).run()
np.savetxt('0002.mo_coeff', mf.mo_coeff, fmt="% 20.16f")
