from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.opt.lbfgs import LBFGS
from quantel.wfn.csf import CSF
from quantel.opt.basin_hopping import BasinHopping
from quantel.opt.hybrid_ef import HybridEF
import numpy as np

mol = PySCFMolecule("mol/h6.xyz", "sto-3g", "angstrom", spin=0)
ints = PySCFIntegrals(mol)

wfn = CSF(ints, "+-+-+-")
wfn.initialise(np.random.rand(wfn.nbsf, wfn.nmo))
opt = HybridEF(maxstep=0.5)
#opt = LBFGS(maxstep=0.5, with_transport=True, with_canonical=True)
opt_kwargs = dict(maxit=500, plev=0, index=1)
bh = BasinHopping(temperature=0.01, random_seed=42, stepsize=2, nminima=50, plev=2, check_overlap=True, optimizer=LBFGS(maxstep=0.5, with_transport=True, with_canonical=True), opt_kwargs=opt_kwargs)
bh.run(wfn, nhop=100)
all_minima = bh.final_quench()