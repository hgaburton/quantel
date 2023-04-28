r"""
Phase factors and permutations are necessary evils in electronic structure.
This module takes care of these evils.
"""


def bubbleSort(arr):
    r"""
    Basic structure of the code taken from https://www.geeksforgeeks.org/python-program-for-bubble-sort/
    :param arr:
    :return:
    """
    n = len(arr)
    # optimize code, so if the array is already sorted, it doesn't need
    # to go through the entire process
    swapped = False
    perm = 0  # Number of permutations
    # Traverse through all array elements
    for i in range(n - 1):
        # range(n) also work but outer loop will
        # repeat one time more than needed.
        # Last i elements are already in place
        for j in range(0, n - i - 1):

            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                perm += 1

        if not swapped:
            # if we haven't needed to make a single swap, we
            # can just exit the main loop.
            return perm
    return perm


def get_phase_factor(alpha_idxs, beta_idxs):
    r"""
    From the orbital representation of a determinant [alpha_idxs], [beta_idxs], find the corresponding
    phase factor of said determinant.

    :param orb_rep:
    :return:
    """
    double_occ = list(set(alpha_idxs).intersection(set(beta_idxs)))
    alpha_singly_occ = list(set(alpha_idxs) - set(double_occ))
    beta_singly_occ = list(set(beta_idxs) - set(double_occ))
    cur_order = alpha_singly_occ + beta_singly_occ
    perm = bubbleSort(cur_order)
    if perm % 2 == 0:   # Even permutation
        return 1
    else:
        return -1
