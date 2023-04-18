r"""
This module generates (and localises) orbitals automatically.
For now, this code is specialised towards generating orbitals for
** homonuclear diatomics **
"""

import numpy as np
import itertools
from scipy.linalg import block_diag

def calculate_metric(coeff_a, coeff_b, sao, perm):
    r"""
    Calculate the metric element C.T * S * P * C
    """
    return np.abs(np.linalg.multi_dot([coeff_a.T, sao, perm, coeff_b]))


def pair_orbitals(mo_coeff, sao):
    r"""
    Given a MO coefficient matrix and AO overlap matrix, pair up MOs in
    bonding/antibonding pairs
    """
    nao = mo_coeff.shape[0] // 2
    perm = block_diag(np.identity(nao), -np.identity(nao))
    pairs = []  # This stores the pair of orbital indices as a tuple
    paired_orbs = []    # This stores orbital indices of orbitals which have been paired up
    unpaired_orbs = [i for i in range(mo_coeff.shape[1])]
    for i in range(mo_coeff.shape[1]):  # Iterate over MOs
        if i not in paired_orbs:
            paired_orbs.append(i)
            unpaired_orbs.remove(i)
            metrics = []
            for j, orb_idx in enumerate(unpaired_orbs):
                metric = calculate_metric(mo_coeff[:, i], mo_coeff[:, orb_idx], sao, perm)
                metrics.append(np.abs(metric-1))
            # Here we want to find the overlap closest to 1
            corresponding_orb_idx = unpaired_orbs[metrics.index(min(metrics))]
            paired_orbs.append(corresponding_orb_idx)
            unpaired_orbs.remove(corresponding_orb_idx)
            pairs.append((i, corresponding_orb_idx))
        else:
            pass
    return pairs

def get_homo_lumo_gap(mo_energy, occ_orb_idx, vir_orb_idx):
    r"""Gets the energy gap between bonding and antibonding orbitals
    """
    return mo_energy[vir_orb_idx] - mo_energy[occ_orb_idx]

def localise(b_coeff, ab_coeff):
    r"""
    :param b_coeff: 2D np.ndarray (Nx1) corresponding to bonding orbital coefficient
    :param ab_coeff: 2D np.ndarray (Nx1) corresponding to antibonding orbital coefficient

    :return: 2D np.ndarray (Nx2) corresponding to localised orbitals
    """
    nao = b_coeff.shape[0]
    temp = (b_coeff + ab_coeff) / np.sqrt(2)
    if np.sum(abs(temp[:nao//2])) > np.sum(abs(temp[nao//2:])):  # Localised on the left-side
        l = (b_coeff + ab_coeff) / np.sqrt(2)
        r = (b_coeff - ab_coeff) / np.sqrt(2)
        return l.reshape(l.shape[0], -1), r.reshape(r.shape[0], -1)
    else:
        l = (b_coeff - ab_coeff) / np.sqrt(2)
        r = (b_coeff + ab_coeff) / np.sqrt(2)
        return l.reshape(l.shape[0], -1), r.reshape(r.shape[0], -1)


def check_bonding(pairs, mo_occ, mo_energy, thresh=1e-8):
    cores = []
    bonds = []
    virs = []
    pairs_left = [i for i in range(len(pairs))]
    for i, pair in enumerate(pairs):
        if mo_occ[pair[0]] + mo_occ[pair[1]] == 4:
            cores.append([pair])
            pairs_left.remove(i)
        elif mo_occ[pair[0]] + mo_occ[pair[1]] == 0:
            virs.append([pair])
            pairs_left.remove(i)
        elif mo_occ[pair[0]] + mo_occ[pair[1]] == 2:
            bonds.append([pair])
            pairs_left.remove(i)
        else: # Occupation is 1 or 3
            if i not in pairs_left:  # We have already added this
                pass
            else:   # Find partner
                pairs_left.remove(i)
                ref_ene = mo_energy[pair[0]]
                for _, pair_left in enumerate(pairs_left):
                    ene = mo_energy[pairs[pair_left][0]]
                    if np.isclose(ref_ene, ene, rtol=0, atol=thresh):
                        bonds.append([pair, pairs[pair_left]])
                        pairs_left.remove(pair_left)
    return cores, bonds, virs

def sort_bonds_by_energy(bonds, mo_energy):
    nbonds = len(bonds)
    es = []
    for i, bond in enumerate(bonds):
        e = mo_energy[bond[0][1]] - mo_energy[bond[0][0]]
        es.append(e)
    energies, bonds = zip(*sorted(zip(es, bonds), key=lambda x: x[0]))
    return bonds, energies

def break_bonds(bonds, bond_energies, n, thresh=1e-8):
    r"""
    We want to break n bonds. 
    The bond energies are in ascending order.
    We shall first figure out the number of degeneracies in bond_energies
    Then use a greedy algorithm to figure out which bonds should be broken
    """
    relevant_e = bond_energies[:n]
    #  We need to check if the last energy is part of a degenerate set.
    #  We check the number of occurences in relevant_e and in bond_energies
    #  This will tell us how many possibilities there are
    ref_e = relevant_e[-1]
    q = 0   # The number of these bonds we have considered
    p = 0   # Total number of degenerate bonds
    for i, e in enumerate(relevant_e):
        if np.isclose(e, ref_e, rtol=0, atol=thresh):
            q += 1
    for i, e in enumerate(bond_energies):
        if np.isclose(e, ref_e, rtol=0, atol=thresh):
            p += 1
    if p == q:
        bonds_to_break = bonds[:n]
        bonds_to_remain = bonds[n:]
        return [bonds_to_break], [bonds_to_remain]
    else:
        # These are fixed
        bonds_to_break = list(bonds[:(n-q)])
        bonds_to_remain = list(bonds[(n+p-q):])
        collated_break = []
        collated_remain = []
        # Now we determine which of the degenerate bonds to break
        chosen_orbs = list(itertools.combinations(np.arange(n-q, n-q+p, 1), q))
        for _, chosen_orb in enumerate(chosen_orbs):
            unchosen_orb = list(set(np.arange(n-q, n-q+p, 1)) - set(list(chosen_orb)))
            bonds_chosen_to_break = [bonds[i] for i in list(chosen_orb)]
            bonds_chosen_to_remain = [bonds[i] for i in unchosen_orb]
            collated_break.append(bonds_to_break + bonds_chosen_to_break)
            collated_remain.append(bonds_to_remain + bonds_chosen_to_remain)
        return collated_break, collated_remain

def break_custom_bonds(bonds, to_break_idxs, thresh=1e-8):
    r"""
    to_break_idx is a List of List[int]. 
    [[0], [2]] means we want to create 2 CSFs, one which breaks the 0th and one which breaks the 2nd bond
    """
    collated_break = []
    collated_remain = []
    for i, to_break_idx in enumerate(to_break_idxs):
        collated_break.append([bonds[k] for k in to_break_idx])
        collated_remain.append([bonds[k] for k in range(len(bonds)) if k not in to_break_idx])
    return collated_break, collated_remain

def create_cv_coeffs(mo_coeff, cvs):
    if cvs == []:
        return None
    idxs = []
    for _, cv in enumerate(cvs):
        assert len(cv) == 1
        idxs.append(cv[0][0])
        idxs.append(cv[0][1])
    return mo_coeff[:, idxs]

def create_remain_coeffs(mo_coeff, remains):
    r"""
    Create coeffs corresponding to orbitals which are not doubly occupied in both bonding
    and antibonding orbitals
    """
    b_idxs = []
    ab_idxs = []
    for _, remain in enumerate(remains):
        assert len(remain) == 1
        b_idxs.append(remain[0][0])
        ab_idxs.append(remain[0][1])
    return mo_coeff[:, b_idxs], mo_coeff[:, ab_idxs]

def create_broken_coeffs(mo_coeffs, brokens):
    r"""
    Create coeffs corresponding to broken bonds
    """
    l_coeffs = None
    r_coeffs = None
    for _, broken in enumerate(brokens):
        for _, pair in enumerate(broken):
            l_coeff, r_coeff = localise(mo_coeffs[:, pair[0]], mo_coeffs[:, pair[1]])
            if l_coeffs is None:
                l_coeffs = l_coeff
            else:
                l_coeffs = np.hstack([l_coeffs, l_coeff])
            if r_coeffs is None:
                r_coeffs = r_coeff
            else:
                r_coeffs = np.hstack([r_coeffs, r_coeff])
    return np.hstack([l_coeffs, r_coeffs])
            

def construct_orbitals(mo_coeff, mo_occ, mo_energy, sao, bo, custom=False, to_break_idxs=None, thresh=1e-8):
    r"""
    This is a method to construct an initial guess from RHF orbitals
    1. Takes in RHF orbitals sorted in ascending order by energy
    2. Pair the orbitals up in bonding/antibonding pairs
    3. If both orbitals in a pair are either both occupied or both unoccupied, ignore
    4. We are left with orbital pairs with total occupation of 1, 2, or 3
        a. For orbital pairs with occupation 2 (BO = 1), they correspond to a bond.
        b. For orbital pairs with occupation 1 or 3, we need to find the corresponding orbital pair.
            The pair dissociate as a bond
    5. For each bond, get the bonding-antibonding energy gap (this shows how easily broken the bond is)
    6. We break the bonds up to a certain bond order as given by bo
    """
    pairs = pair_orbitals(mo_coeff, sao)
    cores, bonds, virs = check_bonding(pairs, mo_occ, mo_energy)
    sorted_bonds, sorted_energies = sort_bonds_by_energy(bonds, mo_energy)
    if custom:
        collated_break, collated_remain = break_custom_bonds(sorted_bonds, to_break_idxs)
    else:
        collated_break, collated_remain = break_bonds(sorted_bonds, sorted_energies, len(sorted_bonds)-bo)
    all_coeffs = []
    core_coeffs = create_cv_coeffs(mo_coeff, cores)
    vir_coeffs = create_cv_coeffs(mo_coeff, virs)
    for i in range(len(collated_break)):
        remain_coeffs, remain_ab_coeffs = create_remain_coeffs(mo_coeff, collated_remain[i])
        broken_coeffs = create_broken_coeffs(mo_coeff, collated_break[i]) 
        if vir_coeffs is None:
            new_mo_coeff = np.hstack([core_coeffs, remain_coeffs, broken_coeffs, remain_ab_coeffs])
        else:
            new_mo_coeff = np.hstack([core_coeffs, remain_coeffs, broken_coeffs, remain_ab_coeffs, vir_coeffs])
        all_coeffs.append(new_mo_coeff)
    return all_coeffs   # Each coeff in the list corresponds to a MO coefficient of a CSF object that we should initialise
