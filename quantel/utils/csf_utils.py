import sys, itertools
import numpy as np

def get_coupling_coefficient(Tn, Pn, tn, pn):
    """ Computes the coupling coefficient C_{tn, pn}^{Tn, Pn}
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
        raise RuntimeError("Invalid spin coupling coefficient requested")

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

def get_Tn(occstr):
    """ Get the total spin vector for a given occupation string
            :param occstr:
            :return: tn, Tn
    """
    _tn = np.array([(0.5 if(st=='+') else -0.5) for st in occstr])
    _Tn = np.cumsum(_tn)
    return _tn, _Tn


def get_determinant_coefficient(det, tn, Tn):
    """ Get the coefficient for a given determinant
            :param det:
            :param tn:
            :param Tn:
            :return:
    """
    # Get the total spin vector for the determinant
    pn, Pn = get_Tn(det)
    
    # Get the coupling coefficient
    coeff = np.prod([get_coupling_coefficient(Tn[i], Pn[i], tn[i], pn[i]) for i in range(len(tn))])

    # Get the phase
    phase = 1
    for j, st1 in enumerate(det):
        for st2 in det[j+1:]:
            if(st1=='-' and st2=='+'):
                phase *= -1

    return coeff * phase


def get_csf_vector(csf):
    """ Get the list of determinants and their coefficients for a given CSF
            :param csf:
            :return:
    """   
    # Check CSF vector is valid
    if(len(csf)==0):
        return [''], [1]
    if(csf[0]!='+'):
        raise RuntimeError("Invalid spin coupling pattern")
    # Get the CSF vectors
    tn, Tn = get_Tn(csf)

    # Get the determinant list and corresponding CI vector
    detlist = list(set([''.join(p) for p in itertools.permutations(csf)]))
    detlist.sort()
    civec = [get_determinant_coefficient(det, tn, Tn) for det in detlist]
    civec = civec / np.linalg.norm(civec)

    # Modify determinant list to occupation strings
    detlist = [det.replace('+','a').replace('-','b') for det in detlist]
    return detlist, civec