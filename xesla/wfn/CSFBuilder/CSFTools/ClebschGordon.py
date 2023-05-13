from sympy import S
from sympy.physics.wigner import clebsch_gordan


def get_cg(j1, j2, j, m1, m2, m, analytic=False):
    r"""
    Get Clebsch-Gordon coefficients. Calculated using Sympy.
    :param j1: Spin of state 1
    :param j2: Spin of state 2
    :param j:  Spin of coupled state
    :param m1: Spin projection of state 1
    :param m2: Spin projection of state 2
    :param m:  Spin projection of coupled state
    :param analytic: :bool: if True, return analytic expression for the Clebsch-Gordon coefficient
    :return:   :float: Clebsch-Gordon coefficient
    """
    expr = clebsch_gordan(S(int(2 * j1)) / 2, S(int(2 * j2)) / 2, S(int(2 * j)) / 2, S(int(2 * m1)) / 2,
                          S(int(2 * m2)) / 2, S(int(2 * m)) / 2)
    if analytic:
        return expr
    else:
        return expr.evalf()


def get_general_tensorprod(j1, j2, j, m):
    r"""
    For a target COUPLED spin state of spin quantum number j with spin projection m,
    compute the necessary linear combination needed from states of spins j1 and j2
    :param j1: Spin of state 1
    :param j2: Spin of state 2
    :param j:  Spin of coupled state
    :param m:  Spin projection of coupled state
    :return:   List of List[float, float, float, float] in [j1, m1, j2, m2] of the states required for coupling
    """
    # We shall work in half-integer steps
    j1 = int(2 * j1)
    j2 = int(2 * j2)
    j = int(2 * j)
    m = int(2 * m)
    assert abs(j1 - j2) <= j <= j1 + j2, "Impossible set of spin quantum numbers"
    states_required = []
    for m1 in range(-j1, j1 + 1, 2):  # m goes in integer steps
        for m2 in range(-j2, j2 + 1, 2):
            if m1 + m2 == m:
                states_required.append([j1 / 2, m1 / 2, j2 / 2, m2 / 2])
    return states_required


def take_csf_tensorprod(kets_a, coeffs_a, kets_b, coeffs_b, cg):
    r"""
    Take the tensor product of the kets and cofficients on different sites. Multiply coefficient product by the
    Clebsch-Gordon coefficient.
    :param kets_a:      List of List[int]. List[int] has format: [pf, a, a, a, ..., b, b, ...]. pf = Phase factor,
                        a and b are alpha and beta occupations, respectively (0 for empty, 1 for filled)
    :param coeffs_a:    1D :np.ndarray: Coefficient of ket_a
    :param kets_b:      As kets_a
    :param coeffs_b:    As coeffs_a
    :param cg:          :float: Clebsch-Gordon coefficient
    :return:            List of List[int] of the coupled state
    """
    new_kets = []
    new_coeffs = []
    for a, ket_a in enumerate(kets_a):
        for b, ket_b in enumerate(kets_b):
            na = (len(ket_a)-1) // 2
            nb = (len(ket_b)-1) // 2
            pf = ket_a[0] * ket_b[0]
            new_ket = [pf] + ket_a[1:na+1] + ket_b[1:nb+1] + ket_a[na+1:] + ket_b[nb+1:]
            new_coeff = float(coeffs_a[a] * coeffs_b[b] * cg)
            new_kets.append(new_ket)
            new_coeffs.append(new_coeff)
    return new_kets, new_coeffs


def get_local_g_coupling(norbs, j):
    r"""
    Construct a genealogical coupling pattern naively.
    :param norbs: :int: Number of orbitals
    :param j:     :float: (Takes on integer or half-integer values) Spin quantum number
    :return:      :str: Corresponding to a genealogical coupling branch e.g. ++-- for V CSF
    """
    j = int(2 * j)
    ps = j
    leftovers = norbs - j
    assert leftovers % 2 == 0
    ps += leftovers // 2
    ns = leftovers // 2
    return ps * "+" + ns * "-"