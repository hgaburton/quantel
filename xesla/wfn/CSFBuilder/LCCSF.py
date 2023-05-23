"""This is a general CSF that is a linear combination of other CSFs"""

import numpy as np
import glob
from pyscf import gto, fci
from typing import Tuple, List
from xesla.drivers import from_file, overlap
from xesla.io.config import Config

from .GenericCSF import GenericCSF


class LCCSF(GenericCSF):
    def __init__(self, mol: gto.Mole, stot: float,
                 n_core: int, n_act: int, dir: str, rel_weights: List):
        super().__init__(stot, mol.spin // 2, n_core, n_act, mol.nelectron)
        self.mol = mol

        # Construct linear combination
        self.wfnlist = self.setup_csf(f"{dir}/lconfig}")
        self.lccoeffs = self.normalise(rel_weights)

    # TODO: Check that the directory structure is correct
    def normalise(self, rel_weights):
        r"""
        Given the relative weights rel_weights, we can normalise the CSF linear combination
        """
        assert len(self.wfnlist) == len(rel_weights), "The number of CSFs used is different from the number of weights"
        smat = overlap(self.wfnlist)
        o = np.einsum("p,pq,q", rel_weights, smat, rel_weights)
        return np.array(rel_weights) / np.sqrt(o)

    def setup_csf(self, fconfig):
        r"""Initialise a CSF here"""
        config = Config(fconfig)
        wfnlist = from_file(self.mol, config)
        return wfnlist


