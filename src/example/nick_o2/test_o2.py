r"""
Driver for NOCI codes
Given a .xyz file and config file, create possible MO coefficients at a particular geometry
Each MO coefficient will lead to a new CSF
"""
import sys, re, copy
sys.path.append('../../')
import numpy as np
import scipy
from pyscf import gto, ao2mo, scf
from pygnme import wick, utils, owndata
from csf import csf
from csfs.ConfigurationStateFunctions.DiatomicOrbitalGenerator import construct_orbitals
from gnme.csf_noci import csf_proj
from gnme.write_noci_results import write_matrix
from opt.mode_controlling import ModeControl


def get_reopt_mo_coeff(mycsf):
    opt  = ModeControl(minstep=0.0, rtrust=0.01)
    opt.run(mycsf, thresh=1e-10, maxit=500, index=None)
    return mycsf.mo_coeff

def run_noci(mol):
    mf = scf.ROHF(mol).run()
    nmo, nocc = mf.mo_occ.size, np.sum(mf.mo_occ > 0)
    sao = owndata(mol.intor('int1e_ovlp'))
    nocc = (9,7)

    # Build required matrix elements
    h1e  = owndata(mf.get_hcore())
    h2e  = owndata(ao2mo.restore(1, mf._eri, mol.nao).reshape(mol.nao**2, mol.nao**2))

    # Build CSFs
    bo0 = csf(mol, 1, 4, 4, 0, [0,1,2,3,4,7], [5,6,8,9], '+++-', [0,1,2,3], 'site')
    new_mo = np.genfromtxt(sys.argv[1]) 
    bo0.initialise(new_mo)

    bo1 = csf(mol, 1, 2, 2, 0, [0,1,2,3,4,5,8], [6,7], '++', [0,1], 'site')
    new_mo = np.genfromtxt(sys.argv[2])
    bo1.initialise(new_mo)
    csfs = [bo0, bo1]

    ci = [owndata(csf.csf_info.ci) for csf in csfs]
    mo = [owndata(get_reopt_mo_coeff(csf)) for csf in csfs]
    nact = [csf.ncas for csf in csfs]
    ncore = [csf.ncore for csf in csfs]

    ci.append(owndata(np.array([[1]])))
    mo.append(owndata(mf.mo_coeff))
    nact.append(2)
    ncore.append(7)
    print(ncore)
    print(nact)
    # Build required matrix elements
    # Core Hamiltonian
    h1e  = owndata(mf.get_hcore())
    # ERIs in AO basis
    h2e  = owndata(ao2mo.restore(1, mf._eri, mol.nao).reshape(mol.nao**2, mol.nao**2))

    h, s, w, v = csf_proj(mol, nmo, nocc, sao, h1e, h2e, tuple(ci), tuple(mo), tuple(ncore), tuple(nact))
    return h, s, w, v

def main():
    # Initialise the molecular structure
    mol = gto.M(
    atom=f"O 0 0 0; O 0 0 2.5;",
    basis='sto-3g',
    spin=2,
    unit="Angstrom")

    # Run NOCI
    h, s, w, v = run_noci(mol)
    write_matrix(sys.argv[3], h, "NO-CSF-CI Hamiltonian matrix")
    write_matrix(sys.argv[3], s, "NO-CSF-CI Overlap matrix")
    write_matrix(sys.argv[3], w, "NO-CSF-CI Eigenvalues")
    write_matrix(sys.argv[3], v, "NO-CSF-CI Eigenvectors")

if __name__ == '__main__':
    main()
