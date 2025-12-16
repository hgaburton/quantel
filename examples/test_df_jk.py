import numpy as np
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
from quantel.wfn.rhf import RHF
import datetime

if __name__ == "__main__":
    print("\n===============================================")
    print(f" Testing JK build with density fitting        ")
    print("===============================================")
    # Setup molecule and integrals
    mol  = PySCFMolecule("formaldehyde.xyz", "aug-cc-pvdz", "angstrom")
    ints = PySCFIntegrals(mol)
    df_ints = PySCFIntegrals(mol,with_df=True)
    print(ints)
    print(df_ints)

    # Initialise RHF object for an open-shell singlet state
    wfn = RHF(ints)
    wfn.get_orbital_guess(method="gwh")
    dm = wfn.dens
    # Time the JK builds with and without density fitting
    t0 = datetime.datetime.now()
    JK = ints.build_JK(dm)
    t1 = datetime.datetime.now()
    print("Conventional JK build time: ",(t1-t0).total_seconds())
    t0 = datetime.datetime.now()
    dfJK = df_ints.build_JK(dm)
    t1 = datetime.datetime.now()
    print("Density-fitted JK build time: ",(t1-t0).total_seconds())
    print(f"JK difference norm = {np.linalg.norm(JK - dfJK)}")