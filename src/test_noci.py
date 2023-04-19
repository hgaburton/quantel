r"""
Driver for NOCI codes
Given a .xyz file and config file, create possible MO coefficients at a particular geometry
Each MO coefficient will lead to a new CSF
"""
import sys, re
import numpy as np
import scipy
from pyscf import gto, ao2mo, scf
from pygnme import wick, utils, owndata
from csf import csf
from csfs.ConfigurationStateFunctions.DiatomicOrbitalGenerator import construct_orbitals
from gnme.csf_noci import csf_proj
from gnme.write_noci_results import write_matrix
from opt.mode_controlling import ModeControl

mol = gto.M(atom=f"O 0 0 0",
            basis='6-31g',
            spin=2,
            unit="Angstrom")
mycsf = csf(mol, 1, 2, 2, 0, [0,1,2], [3,4], '++', mo_basis="site")
mycsf.initialise()
print(mycsf.energy)

mf = scf.ROHF(mol).run()
nmo, nocc = mf.mo_occ.size, np.sum(mf.mo_occ > 0)
sao = owndata(mol.intor('int1e_ovlp'))
nocc = (5,3)
ci = owndata(mycsf.csf_info.get_civec())
mo = owndata(mycsf.mo_coeff)
nact = [2]
ncore = [3]
print("nmo: ", nmo)
print("nocc: ", nocc)
    # Build required matrix elements
    # Core Hamiltonian
h1e  = owndata(mf.get_hcore())
    # ERIs in AO basis
h2e  = owndata(ao2mo.restore(1, mf._eri, mol.nao).reshape(mol.nao**2, mol.nao**2))

h, s, w, v = csf_proj(mol, nmo, nocc, sao, h1e, h2e, tuple(ci), tuple(mo), tuple(ncore), tuple(nact))

print(h)

