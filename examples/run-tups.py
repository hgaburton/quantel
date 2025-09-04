import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import numpy as np
from numpy.random import RandomState
import quantel

from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.rhf import RHF
from quantel.opt.lbfgs import LBFGS
from tups import T_UPS
from quantel.opt.linear import Linear

np.set_printoptions(linewidth=10000,precision=6,suppress=True)
prng = RandomState(10)

guess_coeff = np.fromfile("mo_coeff")
guess_coeff = guess_coeff.reshape((6,6),order='F')
guess_t_amp = np.fromfile("t2_amp")
guess_t_amp = guess_t_amp.reshape((3,3),order='F')

mol  = PySCFMolecule("h6.xyz", "sto-3g", "angstrom")
ints = PySCFIntegrals(mol)

opt = LBFGS(with_transport=False,with_canonical=False,prec_thresh=0.1)
lin = Linear(prec=False)
maxit = 10000

# run_label = ['RHF random', 'RHF small var', 'pCCD random', 'pCCD small var', 'pCCD amplitudes small var',]
# run_label = ['RHF', 'pCCD', 'pCCD amplitudes']
run_label = ['pCCD amplitudes small var', 'pCCD amplitudes']
optimisers = {"LS": lin.run_linesearch, "DL": lin.run_dogleg, "LBFGS": opt.run}
opt = "DL"
prec = "full"


if prec == "approx":
    use_prec = True
    approx_prec = True
elif prec == "full":
    use_prec = True
    approx_prec = False
elif prec == "none":
    use_prec = False
    approx_prec = False
else:
    raise "preconditioner not correctly set"




trials = 10

for run_label in run_label:
    wfn = RHF(ints)
    if run_label.startswith("pCCD"):
        wfn.initialise(guess_coeff)
        perf_pair = True
    else:
        wfn.get_orbital_guess(method="gwh")
        LBFGS().run(wfn, plev=0)
        perf_pair = False

    tUPS = T_UPS(wfn, include_doubles=True, approx_prec=approx_prec, use_prec=use_prec, pp=perf_pair, oo=False, layers=2, plev=0)
    if run_label.endswith("random") or run_label.endswith("var"):
        repeat = trials
    else:
        repeat = 1
    
    for isample in range(repeat):
        print(f"Run: {run_label}")
        print(f"Use preconditioner: {tUPS.use_prec}")
        print(f"Approximate preconditioner: {tUPS.approx_prec}")
        print(f"Orbital Optimised: {tUPS.orb_opt}")
        print(f"Perfect Pairing: {tUPS.perf_pair}")
        print(f"Layers: {tUPS.layers}")
        if run_label.endswith("var"):
            tUPS.x = 0.1*(prng.rand(tUPS.dim) - 0.5)
            print("Random start: [-0.05,0.05]")
        elif run_label.endswith("random"):
            tUPS.x = 2*np.pi*(prng.rand(tUPS.dim) - 0.5)
            print("Random start: (-pi,pi]")
        else:
            tUPS.x = np.zeros(tUPS.dim)
            print("Random start: None")

        if run_label.startswith("pCCD amplitudes"):
            tUPS.x[0] = np.arctan(guess_t_amp[0,0])/tUPS.layers
            tUPS.x[2] = np.arctan(guess_t_amp[1,1])/tUPS.layers
            tUPS.x[4] = np.arctan(guess_t_amp[2,2])/tUPS.layers
            tUPS.x[7] = np.arctan(-guess_t_amp[0,1])/tUPS.layers
            tUPS.x[10] = np.arctan(-guess_t_amp[1,2])/tUPS.layers
            if tUPS.layers >= 2:
                tUPS.x[13] = np.arctan(guess_t_amp[0,0])/tUPS.layers
                tUPS.x[16] = np.arctan(guess_t_amp[1,1])/tUPS.layers
                tUPS.x[19] = np.arctan(guess_t_amp[2,2])/tUPS.layers
                tUPS.x[22] = np.arctan(-guess_t_amp[0,1])/tUPS.layers
                tUPS.x[25] = np.arctan(-guess_t_amp[1,2])/tUPS.layers
                tUPS.x[10] = np.arctan(-guess_t_amp[1,2])/tUPS.layers
            if tUPS.layers >= 3:
                tUPS.x[28] = np.arctan(guess_t_amp[0,0])/tUPS.layers
                tUPS.x[31] = np.arctan(guess_t_amp[1,1])/tUPS.layers
                tUPS.x[34] = np.arctan(guess_t_amp[2,2])/tUPS.layers
                tUPS.x[37] = np.arctan(-guess_t_amp[0,1])/tUPS.layers
                tUPS.x[40] = np.arctan(-guess_t_amp[1,2])/tUPS.layers

        tUPS.update()
        init_x = tUPS.x.copy()
        init_energy = tUPS.energy

        iterations, conv_energy = optimisers[opt](tUPS, maxit=maxit, plev=1)
        print("Initial x:\n", init_x)
        print("Final x:\n", tUPS.x)

        print("--------------BREAK--------------")

        # with open(f"../dump/h6_mol/H6_2.00/final-analysis/layers-1/data-{opt}-{prec}.csv", "a") as f:
        #     f.write(f"{iterations},{conv_energy},{init_energy},{run_label}\n")



