import sys
import numpy as np


def get_coupling_coefficient(Tn, Pn, tn, pn):
    r"""
    Computes the coupling coefficient C_{tn, pn}^{Tn, Pn}
    :param Tn:
    :param Pn:
    :param tn:
    :param pn:
    :return:
    """
    # This is a forbidden case
    if Tn < np.abs(Pn):
        return 0
    if np.isclose(0.5, tn, rtol=0, atol=1e-10):
        return np.sqrt((Tn + 2 * pn * Pn) / (2 * Tn))
    elif np.isclose(-0.5, tn, rtol=0, atol=1e-10):
        return -2 * pn * np.sqrt((Tn + 1 - 2 * pn * Pn) / (2 * (Tn + 1)))
    else:
        print("A coefficient requested is invalid. Exiting.")
        sys.exit(1)


def get_total_coupling_coefficient(det, csf):
    r"""
    Gets the overlap between the determinant and the CSF. This is the coefficient d of the determinant in the CSF.
    :param det:
    :param csf:
    :return:
    """
    total_coeff = 1
    assert len(det) == len(csf), "Number of orbitals in determinant and CSF are not the same. Check again."
    for i in range(1, len(det)):
        Tn = csf[i]
        Pn = det[i]
        tn = csf[i] - csf[i - 1]
        pn = det[i] - det[i - 1]
        total_coeff = total_coeff * get_coupling_coefficient(Tn, Pn, tn, pn)
    return total_coeff