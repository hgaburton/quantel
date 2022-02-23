#!/usr/bin/python

import numpy #as np
import pyscf
from functools import reduce
import scipy.linalg

from pyscf import gto, scf, fci
from pyscf import mcscf
from pyscf.mcscf import addons
from pyscf.fci import addons

import NR_CASSCF
from NR_CASSCF import *
from analyze_result import get_NR_info

ehf = []
emc = []

def run(file, nb, r, ci, mo):
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ["Li", (0., 0., 0.)],           # Geometry corresponding to the grid calculation
        ["F", (0., 0., r)],]
    mol.basis = 'sto-3g'
    mol.unit = 'B'
    mol.build()

    myhf = scf.RHF(mol)
    ehf.append(myhf.scf())

    mycas = NR_CASSCF(myhf,2,2,frozen=0)

    nrj, index, spin, nb_it, time_tot, MO, CI, NO, DM1, nrj, CIno, warn = get_NR_info(file,nb)
    fileCI = np.array(CI)
    fileMO = np.array(MO)


    if mo is None:
        mo = fileMO
        print(fileCI,fileMO)
    else:
        mo = mcscf.project_init_guess(mycas, mo)
    if ci is None:
        ci = fileCI

    mycas._initMO = mo
    mycas._initCI = ci

    mycas.kernel()
    emc.append(mycas.e_tot)
    return mycas.mat_CI, mycas.mo_coeff

# file = "results/lih/mcscf.lih_frozen0_sto-3g.run"
# file = "results/lih/6-31g/mcscf.lih_frozen1_6-31g.run"
# file = "/home/antoinem/PLR1/pyscf/results/lif/sto-3g/mcscf.lif_frozen3.run"
# file = "/home/antoinem/PLR1/pyscf/results/lif/PES_STO3G/mcscf.lif_r5_sto3g_frozen3.run"
# file = "results/h2_6-31g/h2_r0.1_6-31g_fullgrid.txt"
# file = "/home/antoinem/PLR1/pyscf/results/lih/ccpvdz/mcscf.lih_frozen1_ccpvdz.run"
# file = "/home/antoinem/PLR1/pyscf/results/lih/ccpvdz/mcscf.lih_r12_frozen1_ccpvdz.run"
file = "/home/antoinem/PLR1/pyscf/results/lif/sto-3g/mcscf.lif_r5_sto3g_frozen3_15kpts.run"


nb = 5411           # Number of the calculation that we want to use as a starting point for the PES
ci = mo = None

tmp = list(np.arange(3., 5.01, 0.1))
tmp.reverse()

for r in tmp:
    ci, mo = run(file, nb, r, ci, mo)       # Compute the part of the PES before the starting point which correspond to the value of R in the grid calculation (here R=5)

emc.reverse()
print(emc)
emc.pop(-1)
print(emc)

ci = mo = None
for r in np.arange(5, 12.01, 0.1):          # Compute the part of the PES after the starting point
    ci, mo = run(file, nb, r, ci, mo)
print(emc)








# This part is used for the SA-CASSCF PES


# def run(b, dm, mo):
#     mol = gto.Mole()
#     mol.verbose = 0
#     mol.atom = [
#         ["H", (0., 0., 0.)],
#         ["H", (0., 0., b)],]
#     mol.unit = 'B'
#     mol.basis = '6-311g'
#     mol.build()
#     mf = scf.RHF(mol)
#     ehf.append(mf.scf())
#
#     # mc = mcscf.CASSCF(mf, 2, 2)
#
#     mc = mcscf.state_average_(mcscf.CASSCF(mf, 2, 2), [1/5,1/5,1/5,1/5,1/5])
#
#     if mo is None:
#         mo = None
#     else:
#         mo = mcscf.project_init_guess(mc, mo)
#
#     # e1 = mc.mc1step(mo)[0]
#     # emc.append(e1)
#
#     mc.run(mo)
#     emc.append([b]+list(mc.e_states))
#
#     return mf.make_rdm1(), mc.mo_coeff
#
# dm = mo = None
# for b in numpy.arange(2, 12.01, 0.1):
#     dm, mo = run(b, dm, mo)
#
# print(ehf)
# print(emc)













# This part is used for the FCI PES


# list_geom = []
# for i in range(2):
#     tmp_geom = "H 0 0 0; H 0 0 " + str(np.around(8+i*0.1,3))
#     list_geom.append(tmp_geom)
#
# print(list_geom)
#
# listE = []
# it = 0
# for geom in list_geom:
#     mol = gto.Mole()
#     mol.atom = geom
#     mol.unit = 'B'
#     mol.basis = "6-31g"
#     mol.charge = 0
#     mol.spin = 0
#     mol.build()
#     myhf = mol.RHF().run()
#     print(myhf.mo_coeff.tolist())
#
#     nelec = mol.nelec
#     norb = len(myhf.mo_coeff[0])
#
#     fs = fci.addons.fix_spin_(fci.FCI(mol, myhf.mo_coeff), 0)
#     fs.nroots = 2
#     e, c = fs.kernel(verbose=0)
#     print(e)
#     for i, x in enumerate(c):
#         print('state %d, E = %.12f  2S+1 = %.7f' %
#             (i, e[i], fci.spin_op.spin_square0(x, norb, nelec)[1]))
#
#     # listE.append([2+it*0.1]+list(mc.e_states))
#     listE.append([2+it*0.1]+list(e))
#     it+=1
# print(listE)









# In this part I have followed the lowest negative eigenvalue of the lowest LiF singlet at R=5 to obtain the state which behave "diabatically"
# Unfortunately this script is not general and it would need further work to generalize it to follow the lowest eigenvalue of any solution


# file = "/home/antoinem/PLR1/pyscf/results/lif/sto-3g/mcscf.lif_r5_sto3g_frozen3_15kpts.run"
# nb = 5411
#
# nrj, index, spin, nb_it, time_tot, MO, CI, NO, DM1, nrj, CIno, warn = get_NR_info(file,nb)
# fileCI = np.array(CI)
# fileMO = np.array(MO)
#
# mol = gto.Mole()
# mol.verbose = 0
# mol.atom = [
#     ["Li", (0., 0., 0.)],
#     ["F", (0., 0., 5)],]
# mol.basis = 'sto-3g'
# mol.unit = 'B'
# mol.build()
#
# myhf = scf.RHF(mol)
# myhf.scf()
#
# mycas = NR_CASSCF(myhf,2,2,frozen=0)
#
# mycas._initMO = fileMO
# mycas._initCI = fileCI
#
# mycas.kernel()
#
# H_fci = mycas.fcisolver.pspace(mycas.h1eff, mycas.h2eff, mycas.ncas, mycas.nelecas, np=1000000)[1]
# hess = mycas.get_hessian(H_fci)
#
# eigenvalue, eigenvector = scipy.linalg.eig(hess)
#
# eigenvalue = np.around(eigenvalue.real,7)
# print("This is the eigenvalues of the Hessian")
# print(eigenvalue)
# print(eigenvalue[17])
#
# print(eigenvector)
# print(eigenvector[:,17])
#
# print(np.allclose(np.dot(hess,eigenvector[:,17]),eigenvector[:,17]*eigenvalue[17]))
#
# dm1_cas, dm2_cas = mycas.get_CASRDM_12(mycas.mat_CI[:,0])
# g_orb = mycas.get_gradOrb(dm1_cas, dm2_cas)
# g_ci = mycas.get_gradCI(H_fci)
# g = mycas.form_grad(g_orb,g_ci)
#
# nIndepRot = len(g) - len(g_ci)
#
# NR = -0.005*np.dot(scipy.linalg.pinv(hess),eigenvector[:,17])
# NR_Orb = NR[:nIndepRot]
# NR_Orb = mycas.unpack_uniq_var(NR_Orb)
#
# NR_CI = NR[nIndepRot:]
# S = np.zeros((mycas.nDet,mycas.nDet))
# for k in range(1,mycas.nDet):
#     for i in range(mycas.nDet):
#         for j in range(mycas.nDet):
#             S[i,j] += NR_CI[k-1]*(mycas.mat_CI[i,k]*mycas.mat_CI[j,0] - mycas.mat_CI[j,k]*mycas.mat_CI[i,0])
#
# mycas.mo_coeff = mycas.rotateOrb(NR_Orb)
# mycas.mat_CI = mycas.rotateCI(S)
#
# tmp_MO = mycas.mo_coeff
# tmp_CI= mycas.mat_CI
#
# mycas._initMO = mycas.mo_coeff
# mycas._initCI = mycas.mat_CI
#
# mycas.kernel()
#
# ci = mycas.mat_CI
# mo = mycas.mo_coeff
#
# tmp = list(np.concatenate((np.arange(2., 3.96, 0.05),np.arange(4, 5.01, 0.1))))
# print(tmp)
# tmp.reverse()
#
# for r in tmp:
#     ci, mo = run(file, nb, r, ci, mo)
#
# emc.reverse()
# print(emc)
# emc.pop(-1)
# print(emc)
#
# ci = tmp_CI
# mo = tmp_MO
#
# for r in np.arange(5, 12.01, 0.1):
#     ci, mo = run(file, nb, r, ci, mo)
# print(emc)




