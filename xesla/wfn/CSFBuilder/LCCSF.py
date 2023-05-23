"""This is a general CSF that is a linear combination of other CSFs"""

import numpy as np
import glob
from pyscf import gto, fci
from typing import Tuple, List
from xesla.drivers import overlap
from xesla.io.config import Config

from .GenericCSF import GenericCSF


class LCCSF(GenericCSF):
    def __init__(self, mol: gto.Mole, stot: float,
                 n_core: int, n_act: int, dir: str, rel_weights: List):
        super().__init__(stot, mol.spin // 2, n_core, n_act, mol.nelectron)
        self.mol = mol

        # Construct linear combination
        self.wfnlist = self.setup_csf(f"{dir}/lconfig")
        self.lccoeffs = self.normalise(rel_weights)
        self.rdm1, self.rdm2 = self.get_rdms()
        self.ci = self.get_ci()

    def lc_from_file(self, config):
        from xesla.wfn.csf import CSF as WFN
        ndet = 0

        # Initialise wavefunction list
        wfn_list = []
        wfnconfig = config["wavefunction"]["csf"]

        count = 0
        for prefix in config["jobcontrol"]["read_dir"]:
            print(" Reading solutions from directory {:s}".format(prefix))
            # Need to count the number of states to converge
            nstates = len(glob.glob(prefix+"*.mo_coeff"))
            for i in range(nstates):
                old_tag = "{:s}{:04d}".format(prefix, i+1)

                try: del myfun
                except: pass
                myfun = WFN(self.mol, **wfnconfig)
                myfun.read_from_disk(old_tag)

                # Deallocate integrals to reduce memory footprint
                myfun.deallocate()
                wfn_list.append(myfun.copy())
        return wfn_list

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
        wfnlist = self.lc_from_file(config)
        return wfnlist

    def get_rdms(self):
        r"""
        Get the RDMs for a multi-reference system
        THIS IS JUST TO SATISFY THE CODE, THE RETURNED VALUES ARE WRONG
        """
        return self.wfnlist[0].csf_instance.rdm1, self.wfnlist[0].csf_instance.rdm2

    def get_ci(self):
        r"""
        Get civector
        THIS IS JUST TO SATISFY THE CODE, THE RETURNED VALUES ARE WRONG
        """
        return self.wfnlist[0].csf_instance.ci

