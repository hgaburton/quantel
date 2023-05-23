#!/usr/bin/python3

import numpy
from .parser import getvalue, getlist, getbool

class Config(dict):
    def __init__(self, fname):
        self.__fname = fname
        self.__parsed = False

        with open(self.__fname, "r") as inF:
            self.lines  = inF.read().splitlines()

        self.parse()

    def print(self):
        print()
        print("------------")
        print("Input File:")
        print("------------")
        with open(self.__fname, "r") as inF:
           print(inF.read().strip())
        print("------------")
        print()

    def parse(self):
        self.parse_molecule()
        self.parse_wavefunction()
        self.parse_optimiser()
        self.parse_jobcontrol()


    def parse_molecule(self):
        """Read keywords that define the molecular system"""
        self["molecule"] = dict(charge = getvalue(self.lines,"charge",int,True),
                                spin = getvalue(self.lines,"spin",int,True),
                                basis = getvalue(self.lines,"basis",str,True), 
                                unit = getvalue(self.lines,"units",str,False,"Ang")
                               )

            
    def parse_wavefunction(self):
        """Read keywords that define the wavefunction model"""
        self["wavefunction"] = dict(method = getvalue(self.lines,"method",str,True).lower())

        # Get keywords for each allowed method
        if self["wavefunction"]["method"] == "esmf":
            # TODO: Add support for triplet states 
            self["wavefunction"]["esmf"] = dict(ref_allowed = getbool(self.lines,"with_ref",False,True))

        elif self["wavefunction"]["method"] == "casscf":
            self["wavefunction"]["casscf"] = dict(active_space = getlist(self.lines,"active_space",int,True))

        elif self["wavefunction"]["method"] == "csf":
            self["wavefunction"]["csf"] = dict(g_coupling = getvalue(self.lines,"genealogical_coupling",str,False),
                                               mo_basis = getvalue(self.lines,"mo_basis",str,False),
                                               active = getlist(self.lines,"active_orbitals",int,False),
                                               active_space=getlist(self.lines,"active_space",int,False),
                                               core = getlist(self.lines,"core_orbitals",int,False),
                                               permutation = getlist(self.lines,"coupling_permutation",int,False),
                                               stot = getvalue(self.lines,"total_spin",float,True),
                                               csf_build = getvalue(self.lines,"csf_build",str,True),
                                               localstots = getlist(self.lines,"local_spins",float,False),
                                               active_subspaces = getlist(self.lines,"active_subspaces",int,False),
                                               lcdir = getvalue(self.lines,"linearcombdir",str,False,None),
                                               rel_weights=getlist(self.lines, "relative_weights", int, False)
                                              )


    def parse_optimiser(self):
        """Read keywords that define the optimiser we use"""
        self["optimiser"] = dict(algorithm = getvalue(self.lines,"algorithm",str,False,default="eigenvector_following").lower())

        if self["optimiser"]["algorithm"] == "eigenvector_following":
            self["optimiser"]["eigenvector_following"] = dict(minstep = getvalue(self.lines,"minstep",float,False,default=0),
                                                              rtrust  = getvalue(self.lines,"rstrust",float,False,default=0.15),
                                                              maxstep = getvalue(self.lines,"maxstep",float,False,default=numpy.pi),
                                                              hesstol = getvalue(self.lines,"hesstol",float,False,1e-16)
                                                             )
        elif self["optimiser"]["algorithm"] == "mode_control":
            self["optimiser"]["mode_control"] = dict(minstep = getvalue(self.lines,"minstep",float,False,default=0),
                                                    rtrust  = getvalue(self.lines,"rstrust",float,False,default=0.15),
                                                    maxstep = getvalue(self.lines,"maxstep",float,False,default=numpy.pi),
                                                    hesstol = getvalue(self.lines,"hesstol",float,False,1e-16)
                                                    )

        else:
            errstr = "Requested optimiser '"+self["optimiser"]["algorithm"]+"' is not available"
            raise ValueError(errstr)


        self["optimiser"]["keywords"] = dict(thresh = getvalue(self.lines,"convergence",float,False,default=1e-8),
                                             maxit = getvalue(self.lines,"maxit",int,False,default=500),
                                             index = getvalue(self.lines,"target_index",int,False,default=None)
                                            )

        

    def parse_jobcontrol(self):
        """Parse keywords that define how jobs are run"""
        self["jobcontrol"] = dict(guess = getvalue(self.lines,"guess",str,False,default="random").lower(), 
                                  noci  = getbool(self.lines,"noci",False,default=False),
                                  dist_thresh = getvalue(self.lines,"dist_tresh",float,False,default=1e-8),
                                  ovlp_mat = getbool(self.lines,"overlap_matrix",False,default=False),
                                  skip_ovlp_check = getbool(self.lines,"skip_ovlp_check",False,default=False)
                                  ) 
        
        if self["jobcontrol"]["guess"] == "random":
            self["jobcontrol"]["search"] = dict(nsample = getvalue(self.lines,"nsample",int,False,default=10),
                                                seed = getvalue(self.lines,"seed",int,False,default=7)
                                               )

        elif self["jobcontrol"]["guess"] == "fromfile":
            self["jobcontrol"]["read_dir"] = getlist(self.lines,"read_dir",str,True)

        elif self["jobcontrol"]["guess"] == "ciguess":
            self["jobcontrol"]["ci_guess"] = getlist(self.lines,"ci_guess",int,True)

        else:
            errstr = "'"+self["jobcontrol"]["guess"]+"' is not a valid option for keyword 'guess'"
            raise ValueError(errstr)

        if self["jobcontrol"]["noci"]:
            self["jobcontrol"]["noci_job"] = dict(lindep_tol = getvalue(self.lines,"lindep_tol",float,False,default=1e-8), 
                                                  plev = getvalue(self.lines,"plevel",int,False,default=1)
                                                 )

