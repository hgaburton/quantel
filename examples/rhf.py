import quantel
import numpy as np
from quantel.wfn.rhf import RHF
from pygnme import utils
import time

# Test RHF object with a range of optimisers
print("Test RHF object with a range of optimisers")

# Setup molecule
mol = quantel.Molecule([["C", -0.0024458196,   0.000000000,    0.000000000],
                        ["O",  1.1818599990,   0.000000000,    0.000000000],
                        ["H", -0.5834770897,   0.000000000,   -0.924107645],
                        ["H", -0.5834770897,   0.000000000,    0.924107645]], 
                        "angstrom")
print("Molecule:")
mol.print()

# Setup integral interface
ints = quantel.LibintInterface("6-31g", mol)

# Initialise RHF object
wfn = RHF(ints)

# Setup optimiser
for guess in ("gwh", "core"):
    print("\n************************************************")
    print(f" Testing '{guess}' initial guess method")
    print("************************************************")
    from quantel.opt.lbfgs import LBFGS
    wfn.get_orbital_guess(method="gwh")
    LBFGS().run(wfn)

    from quantel.opt.diis import DIIS
    wfn.get_orbital_guess(method="gwh")
    DIIS().run(wfn)
