r"""
Write NOCI results to file
"""
import numpy as np
import pandas as pd

def write_matrix(ofile: str, mat: np.ndarray, title: str):
    r"""
    This writes the CI Hamiltonian matrix into an output file ofile
    :return:
    """
    with open(ofile, "a") as f:
        f.write(
            '***************************************************************************************************\n')
        f.write(title + '\n')
        f.write(
            '***************************************************************************************************\n')
        df = pd.DataFrame(np.round(mat, 6))
        dfstr = df.to_string(header=False, index=False)
        f.write(dfstr)
        f.write("\n")

