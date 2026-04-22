#!/usr/bin/python

import numpy as np

def write_fcidump(integrals,mo_coeff:np.ndarray,filename:str="FCIDUMP",thresh=1e-15):
    """Write an FCIDUMP file from any integral object.

    Parameters
    ----------
    integrals : integral object
        Any object exposing ``molecule()``, ``scalar_potential()``,
        ``oei_ao_to_mo()``, and ``tei_ao_to_mo()`` (e.g. PySCFIntegrals or FCIDUMP).
    filename : str
        Output file path.
    mo_coeff : ndarray of shape (nbsf, nmo)
        MO coefficient matrix. Integrals are transformed to the MO basis before writing.
    thresh : float, optional
        Integral values with absolute value below this threshold are omitted.
        Default is 1e-15.
    """
    # Get information from the integrals object
    nalfa = integrals.molecule().nalfa()
    nbeta = integrals.molecule().nbeta()
    nelec = nalfa + nbeta
    ms2 = nalfa - nbeta

    # Get energy components
    C = np.asarray(mo_coeff, dtype=np.float64)
    norb = C.shape[1]
    h = integrals.oei_ao_to_mo(C, C)
    # tei_ao_to_mo returns physicist's <pq|rs>; opposite-spin (alpha1!=alpha2)
    # gives no antisymmetrization. Convert to chemist's (pq|rs) = <pr|qs>
    # via .transpose(0,2,1,3).
    g = integrals.tei_ao_to_mo(C,C,C,C,True,False).transpose(0,2,1,3)

    # Write the output file
    with open(filename, 'w') as f:
        # Header
        orbsym = ','.join(['1'] * norb)
        f.write(f" &FCI NORB={norb:<d}, NELEC={nelec:d}, MS2={ms2:d},\n")
        f.write(f"  ORBSYM={orbsym},\n")
        f.write(f"  ISYM=1,\n")
        f.write(f" &END\n")

        # Two-electron integrals (chemist's notation, 8-fold symmetry, 1-indexed)
        # Loop over unique (pq|rs) with p>=q, r>=s, pq>=rs
        for i in range(norb):
            for j in range(i+1):
                ij = i*norb + j
                for k in range(norb):
                    for l in range(k+1):
                        kl = k*norb + l
                        if ij < kl: continue
                        val = g[i,j,k,l]
                        if abs(val) > thresh:
                            f.write(f"{val:23.16e} {i+1:4d} {j+1:4d} {k+1:4d} {l+1:4d}\n")

        # One-electron integrals (upper triangle, 1-indexed)
        for i in range(norb):
            for j in range(i+1):
                val = h[i,j]
                if abs(val) > thresh:
                    f.write(f"{val:23.16e} {i+1:4d} {j+1:4d} {0:4d} {0:4d}\n")

        # Scalar potential (nuclear repulsion / frozen-core energy)
        f.write(f"{integrals.scalar_potential():23.16e} {0:4d} {0:4d} {0:4d} {0:4d}\n")