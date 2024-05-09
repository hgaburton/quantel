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

def read_noci_config(file):
    r"""
    Reads in a config file for the NOCI calculation
    """
    f = open(file, "r")
    lines = f.read().splitlines()
    basis, charge, spin, bo_list, rhf = 'sto-3g', 0, 0, [], True
    for line in lines:
        if re.match('basis', line) is not None:
            basis = str(re.split(r'\s', line)[-1])
        elif re.match('charge', line) is not None:
            charge = int(re.split(r'\s', line)[-1])
        elif re.match('spin', line) is not None:
            spin = int(re.split(r'\s', line)[-1])
        elif re.match('bond_orders', line) is not None:
            bo_list = [int(x) for x in re.split(r'\s', line)[1:]]
        elif re.match('rhf', line) is not None:
            if re.split(r'\s', line)[-1].lower() == 'true':
                rhf = True
            else:
                rhf = False
    return basis, charge, spin, bo_list, rhf

def read_config(file):
    f = open(file, "r")
    lines = f.read().splitlines()
    basis, charge, spin, frozen, cas, grid_option, Hind, maxit, thresh, core, active, g_coupling, permutation, custom, breaking = 'sto-3g', 0, 0, 0, (
    0, 0), 1000, None, 1000, 1e-8, [], [], None, None, False, None
    nsample = 1
    unit_str = 'A'
    for line in lines:
        if re.match('basis', line) is not None:
            basis = str(re.split(r'\s', line)[-1])
        elif re.match('charge', line) is not None:
            charge = int(re.split(r'\s', line)[-1])
        elif re.match('spin', line) is not None:
            spin = int(re.split(r'\s', line)[-1])
        elif re.match('frozen', line) is not None:
            frozen = int(re.split(r'\s', line)[-1])
        elif re.match('seed', line) is not None:
            np.random.seed(int(line.split()[-1]))
        elif re.match('index', line) is not None:
            Hind = int(line.split()[-1])
        elif re.match('maxit', line) is not None:
            maxit = int(line.split()[-1])
        elif re.match('cas', line) is not None:
            tmp = re.split(r'\s', line)[-1]
            tmp2 = tmp[1:-1].split(',')
            cas = (int(tmp2[0]), int(tmp2[1]))
        elif re.match('nsample', line) is not None:
            nsample = int(re.split(r'\s', line)[-1])
        elif re.match('units', line) is not None:
            unit_str = str(re.split(r'\s', line)[-1])
        elif re.match('thresh', line) is not None:
            thresh = np.power(0.1, int(re.split(r'\s', line)[-1]))
        elif re.match('core', line) is not None:
            core = [int(x) for x in re.split(r'\s', line)[1:]]
        elif re.match('active', line) is not None:
            active = [int(x) for x in re.split(r'\s', line)[1:]]
        elif re.match('g_coupling', line) is not None:
            g_coupling = line.split()[-1]
        elif re.match('permutation', line) is not None:
            permutation = [int(x) for x in re.split(r'\s', line)[1:]]
        elif re.match('custom', line) is not None:
            if re.split(r'\s', line)[-1].lower() == 'true':
                custom = True
            else:
                custom = False
        elif re.match('breaking', line) is not None:
            breaking = []
            tmp = re.split(r'\s', line)[-1]
            tmp_bonds = tmp.split(';')
            for tmp_bond in tmp_bonds:
                bonds = tmp_bond.split(',')
                int_bonds = [int(x) for x in bonds]
                breaking.append(int_bonds)
    return basis, charge, spin, frozen, cas, nsample, Hind, maxit, unit_str, thresh, core, active, g_coupling, permutation, custom, breaking

def get_csfs(mol, mo_coeff, mo_occ, mo_energy, sao, bo_list, keep_rhf=True, thresh=1e-8):
    r"""
    Get relevant CSFs by initialising with the MO coefficients obtained
    :param mol: gto.Mole object
    :param bo_list: List[int] List of different bond orders considered
    """
    csfs = []
    for i, bo in enumerate(bo_list):
        basis, charge, spin, frozen, cas, nsample, Hind, maxit, unit_str, thresh, core, active, g_coupling, permutation, custom, breaking = read_config(f"{bo}_config.txt")
        bo_coeffs = construct_orbitals(mo_coeff, mo_occ, mo_energy, sao, bo, custom, breaking, thresh)
        for j, coeff in enumerate(bo_coeffs):
            mycsf = csf(mol, spin, cas[0], cas[1], frozen, core, active, g_coupling, permutation)
            mycsf.initialise(coeff)
            csfs.append(mycsf)
    # Hacking this
    #hacked_csf = csf(mol, 0, 2, 2, 0, [0,1,2,3,5,6], [4,9], '+-', [0,1], 'site')
    #hacked_csf.initialise()
    #from opt.eigenvector_following import EigenFollow
    #opt = EigenFollow(minstep=0.0, rtrust=0.15)
    #opt.run(hacked_csf, thresh=1e-8, maxit=500, index=0)
    #csfs.append(hacked_csf)
    return csfs

def get_reopt_mo_coeff(mycsf):
    opt  = ModeControl(minstep=0.0, rtrust=0.01)
    opt.run(mycsf, thresh=1e-10, maxit=500, index=None)
    return mycsf.mo_coeff

def run_noci(mol, bo_list, keep_rhf=True):
    mf = scf.RHF(mol).run()
    nmo, nocc = mf.mo_occ.size, np.sum(mf.mo_occ > 0)
    sao = owndata(mol.intor('int1e_ovlp'))

    csfs = get_csfs(mol, mf.mo_coeff, mf.mo_occ, mf.mo_energy, sao, bo_list)
    ci = [owndata(csf.csf_info.ci) for csf in csfs]
    mo = [owndata(get_reopt_mo_coeff(csf)) for csf in csfs]
    nact = [csf.ncas for csf in csfs]
    ncore = [csf.ncore for csf in csfs]

    for i, csf in enumerate(csfs):
        np.savetxt(f"{i}_opt.mo_coeff", csf.mo_coeff, fmt="% 20.16f")

    if keep_rhf:
        ci.append(np.array([[1]]))
        mo.append(mf.mo_coeff)
        nact.append(0)
        ncore.append(mol.nelectron // 2)

    # Build required matrix elements
    # Core Hamiltonian
    h1e  = owndata(mf.get_hcore())
    # ERIs in AO basis
    h2e  = owndata(ao2mo.restore(1, mf._eri, mol.nao).reshape(mol.nao**2, mol.nao**2))

    h, s, w, v = csf_proj(mol, nmo, nocc, sao, h1e, h2e, tuple(ci), tuple(mo), tuple(ncore), tuple(nact))
    return h, s, w, v

def main():
    # Initialise the molecular structure
    basis, charge, spin, bo_list, rhf = read_noci_config(sys.argv[2])
    mol = gto.Mole(symmetry=False, unit='A')
    mol.atom = sys.argv[1]
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()
    # Run NOCI
    h, s, w, v = run_noci(mol, bo_list, rhf)
    write_matrix(sys.argv[3], h, "NO-CSF-CI Hamiltonian matrix")
    write_matrix(sys.argv[3], s, "NO-CSF-CI Overlap matrix")
    write_matrix(sys.argv[3], w, "NO-CSF-CI Eigenvalues")
    write_matrix(sys.argv[3], v, "NO-CSF-CI Eigenvectors")

if __name__ == '__main__':
    main()
