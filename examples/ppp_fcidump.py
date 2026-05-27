import numpy as np

eparams = dict(C=0.0, N1=-2.96, N2=-17.56, Cl=-12.65)
zparams = dict(C=1.0, N1=1.0, N2=1.0, Cl=1.0)
Uparams = dict(C=8.0, N1=12.34, N2=16.76, Cl=8.0)
r0params = dict(C=1.328, N1=1.115, N2=1.115, Cl=1.987)
bparams = dict(C=dict(C=1.66,N1=1.66,N2=1.66,Cl=1.66))
Aparams = dict(C=dict(C=-28.08,N1=-23.54,N2=-22.16,Cl=-27.10))

def write_ppp_fcidump(atoms, bonds=None,
                      filename='PPP.FCIDUMP',
                      dihedral_angles=None,
                      atom_params=None, hop_params=None,
                      n_elec_override=None, ms2=0,
                      units='hartree', thresh=1e-15):

    # Atoms is a list of (atom_type, position) tuples.  
    # First get the relevant arrays
    natom = len(atoms)
    print(atoms)
    Ep = np.array([eparams[a[0]] for a in atoms])
    Zp = np.array([zparams[a[0]] for a in atoms])
    rp = np.array([r0params[a[0]] for a in atoms])
    Up = np.array([Uparams[a[0]] for a in atoms])

    # Get two-electron arrays
    Apq = np.array([[Aparams[a[0]][b[0]] for b in atoms] for a in atoms])
    Bpq = np.array([[bparams[a[0]][b[0]] for b in atoms] for a in atoms])
    Upq = 0.5 * (Up[:, None] + Up[None, :])
    rpq = 0.5 * (rp[:, None] + rp[None, :])
    print(Apq)
    print(Upq)

    # Get array of dihedral angles




    # Step 1: process molecular structure
    sites, bonds = process_molecule(atom_types, positions, bonds)
    N = len(sites)

    # Step 2 & 3: build parameter arrays
    epsilon, Z, A_mat, b_mat, U_mat, r0_mat = build_parameter_arrays(
        sites, atom_params, hop_params
    )

    # Build Hamiltonian
    h1e, h2e, E_nuc, n_elec = build_ppp_hamiltonian(
        sites, bonds, epsilon, Z, A_mat, b_mat, U_mat, r0_mat, dihedral_angles
    )

    if n_elec_override is not None:
        n_elec = n_elec_override

    nalfa = (n_elec + ms2) // 2
    nbeta =  n_elec - nalfa
    norb  = N

    # Unit conversion (eV → Hartree by default)
    if units.lower() == 'hartree':
        h1e   = h1e   * _HARTREE_PER_EV
        h2e   = h2e   * _HARTREE_PER_EV
        E_nuc = E_nuc * _HARTREE_PER_EV
        unit_label = 'Hartree'
    else:
        unit_label = 'eV'

    print(f"  Writing {filename}")
    print(f"  Sites={norb},  N_elec={n_elec} (α={nalfa}, β={nbeta}),  ms2={ms2}")
    print(f"  E_nuc = {E_nuc:.8f} {unit_label}")

    with open(filename, 'w') as f:
        orbsym = ','.join(['1'] * norb)
        f.write(f" &FCI NORB={norb:d}, NELEC={n_elec:d}, MS2={ms2:d},\n")
        f.write(f"  ORBSYM={orbsym},\n")
        f.write(f"  ISYM=1,\n")
        f.write(f" &END\n")

        # Two-electron integrals in 8-fold-reduced chemist's notation.
        # Under ZDO only (ii|kk) = γ_ik are non-zero, so the unique entries
        # are γ_ik for i ≥ k.
        for i in range(norb):
            for j in range(i + 1):
                ij = i * norb + j
                for k in range(norb):
                    for l in range(k + 1):
                        kl = k * norb + l
                        if ij < kl:
                            continue
                        val = h2e[i, j, k, l]
                        if abs(val) > thresh:
                            f.write(f"{val:23.16e} {i+1:4d} {j+1:4d} {k+1:4d} {l+1:4d}\n")

        # One-electron integrals (upper triangle)
        for i in range(norb):
            for j in range(i + 1):
                val = h1e[i, j]
                if abs(val) > thresh:
                    f.write(f"{val:23.16e} {i+1:4d} {j+1:4d} {0:4d} {0:4d}\n")

        # Scalar constant (nuclear-repulsion analog)
        f.write(f"{E_nuc:23.16e} {0:4d} {0:4d} {0:4d} {0:4d}\n")

    print(f"  Done.")
    return h1e, h2e, E_nuc, n_elec


# ---------------------------------------------------------------------------
# Example molecules (π-centre coordinates only, in Ångström)
# ---------------------------------------------------------------------------

def ethylene_geometry():
    """Ethylene — simplest PPP system (2 π-sites)."""
    positions  = np.array([[0.00, 0.0, 0.0],
                            [1.34, 0.0, 0.0]])
    atom_types = ['C', 'C']
    bonds      = [(0, 1)]
    return list(zip(atom_types,positions))


def benzene_geometry():
    """Benzene — 6 π-sites on a regular hexagon (C-C = 1.40 Å)."""
    R      = 1.40   # circumradius = bond length for a regular hexagon
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    positions  = np.column_stack([R * np.cos(angles),
                                  R * np.sin(angles),
                                  np.zeros(6)])
    atom_types = ['C'] * 6
    bonds      = [(i, (i + 1) % 6) for i in range(6)]
    return atom_types, positions, bonds


def naphthalene_geometry():
    """Naphthalene — 10 π-sites (two fused rings, C-C ≈ 1.40 Å)."""
    positions = np.array([
        [ 0.000,  1.400, 0.0],   # 0
        [ 1.212,  0.700, 0.0],   # 1
        [ 1.212, -0.700, 0.0],   # 2
        [ 0.000, -1.400, 0.0],   # 3
        [-1.212, -0.700, 0.0],   # 4  bridgehead
        [-1.212,  0.700, 0.0],   # 5  bridgehead
        [-2.424,  1.400, 0.0],   # 6
        [-3.636,  0.700, 0.0],   # 7
        [-3.636, -0.700, 0.0],   # 8
        [-2.424, -1.400, 0.0],   # 9
    ])
    atom_types = ['C'] * 10
    bonds = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),
             (4,9),(9,8),(8,7),(7,6),(6,5)]
    return atom_types, positions, bonds


# ---------------------------------------------------------------------------
# Main: generate FCIDUMP files for the example molecules
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  PPP FCIDUMP generator")
    print("=" * 60)

    print("\n-- Ethylene --")
    write_ppp_fcidump(ethylene_geometry(), filename='ethylene_ppp.fcidump')

    print("\n-- Benzene --")
    write_ppp_fcidump(*benzene_geometry(), filename='benzene_ppp.fcidump')

    print("\n-- Naphthalene --")
    write_ppp_fcidump(*naphthalene_geometry(), filename='naphthalene_ppp.fcidump')

    print("\n-- Pyridine (pyridine-type N at site 0) --")
    R      = 1.39
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    pyr_pos   = np.column_stack([R * np.cos(angles), R * np.sin(angles), np.zeros(6)])
    pyr_types = ['N1', 'C', 'C', 'C', 'C', 'C']
    pyr_bonds = [(i, (i + 1) % 6) for i in range(6)]
    write_ppp_fcidump(pyr_types, pyr_pos, pyr_bonds, filename='pyridine_ppp.fcidump')

    print("\n-- Benzyl radical (doublet, ms2=1) --")
    types, pos, bonds = benzene_geometry()
    benz_pos   = np.vstack([pos, [[2.50, 0.0, 0.0]]])
    benz_types = types + ['C']
    benz_bonds = bonds + [(0, 6)]
    write_ppp_fcidump(benz_types, benz_pos, benz_bonds,
                      filename='benzyl_ppp.fcidump', ms2=1)

    print("\n" + "=" * 60)
    print("  All FCIDUMP files written.")
    print("=" * 60)
