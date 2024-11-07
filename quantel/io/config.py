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
        self["molecule"] = dict(basis = getvalue(self.lines,"basis",str,True), 
                                unit = getvalue(self.lines,"units",str,False,"angstrom")
                               )
        if (self["molecule"]["unit"] == "ang") or (self["molecule"]["unit"] == "a"):
            self["molecule"]["unit"] = "angstrom"

            
    def parse_wavefunction(self):
        """Read keywords that define the wavefunction model"""
        self["wavefunction"] = dict(method = getvalue(self.lines,"method",str,True).lower())

        # Get keywords for each allowed method
        if self["wavefunction"]["method"] == "esmf":
            # TODO: Add support for triplet states 
            self["wavefunction"]["esmf"] = dict(with_ref = getbool(self.lines,"with_ref",False,True))
        
        elif self["wavefunction"]["method"] == "pp":
            self["wavefunction"]["pp"] = dict()

        elif self["wavefunction"]["method"] == "rhf":
            self["wavefunction"]["rhf"] = dict()

        elif self["wavefunction"]["method"] == "casscf":
            self["wavefunction"]["casscf"] = dict(active_space = getlist(self.lines,"active_space",int,True))

        elif self["wavefunction"]["method"] == "csf":
            self["wavefunction"]["csf"] = dict(spin_coupling = getvalue(self.lines,"genealogical_coupling",str,True))


    def parse_optimiser(self):
        """Read keywords that define the optimiser we use"""
        self["optimiser"] = dict(algorithm = getvalue(self.lines,"algorithm",str,False,default="eigenvector_following").lower())

        if self["optimiser"]["algorithm"] == "eigenvector_following":
            self["optimiser"]["eigenvector_following"] = dict(minstep = getvalue(self.lines,"minstep",float,False,default=0),
                                                              rtrust  = getvalue(self.lines,"rtrust",float,False,default=0.15),
                                                              maxstep = getvalue(self.lines,"maxstep",float,False,default=0.2),
                                                              hesstol = getvalue(self.lines,"hesstol",float,False,1e-16)
                                                             )
        elif self["optimiser"]["algorithm"] == "mode_control":
            self["optimiser"]["mode_control"] = dict(minstep = getvalue(self.lines,"minstep",float,False,default=0),
                                                    rtrust  = getvalue(self.lines,"rtrust",float,False,default=0.15),
                                                    maxstep = getvalue(self.lines,"maxstep",float,False,default=numpy.pi),
                                                    hesstol = getvalue(self.lines,"hesstol",float,False,1e-16)
                                                    )
        elif self["optimiser"]["algorithm"] == "gmf":
            self["optimiser"]["gmf"] = dict(minstep = getvalue(self.lines,"minstep",float,False,default=0),
                                            maxstep = getvalue(self.lines,"maxstep",float,False,default=0.2),
                                            with_transport = getbool(self.lines,"parallel_transport",False,default=True),
                                            with_canonical = getbool(self.lines,"pseudo-canonicalise",False,default=True),
                                            canonical_interval = getvalue(self.lines,"canonical_interval",int,False,default=10),
                                            max_subspace = getvalue(self.lines,"max_subspace",int,False,default=10)
                                            )
        elif self["optimiser"]["algorithm"] == "lsr1":
            self["optimiser"]["lsr1"] = dict(minstep = getvalue(self.lines,"minstep",float,False,default=0),
                                            rtrust  = getvalue(self.lines,"rtrust",float,False,default=0.15),
                                            maxstep = getvalue(self.lines,"maxstep",float,False,default=0.2),
                                            max_subspace = getvalue(self.lines,"max_subspace",int,False,default=10),
                                            precmin = getvalue(self.lines,"precmin",float,False,default=1)
                                            )
        elif self["optimiser"]["algorithm"] == "lbfgs":
            self["optimiser"]["lbfgs"] = dict(minstep = getvalue(self.lines,"minstep",float,False,default=0),
                                            maxstep = getvalue(self.lines,"maxstep",float,False,default=0.2),
                                            max_subspace = getvalue(self.lines,"max_subspace",int,False,default=10),
                                            backtrack_scale = getvalue(self.lines,"backtrack_scale",float,False,default=0.1),
                                            with_transport = getbool(self.lines,"parallel_transport",False,default=True),
                                            with_canonical = getbool(self.lines,"pseudo-canonicalise",False,default=True),
                                            canonical_interval = getvalue(self.lines,"canonical_interval",int,False,default=10),
                                            gamma_preconditioner = getbool(self.lines,"gamma_prec",False,default=False)
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
                                  oscillator_strength = getbool(self.lines,"oscillator_strength",False,default=False),
                                  dist_thresh = getvalue(self.lines,"dist_tresh",float,False,default=1e-8),
                                  ovlp_mat = getbool(self.lines,"overlap_matrix",False,default=False),
                                  analyse = getbool(self.lines,"analyse",False,default=False), 
                                  nevpt2 = getbool(self.lines,"nevpt2",False,default=False)
                                 ) 
        
        if self["jobcontrol"]["guess"] == "random":
            self["jobcontrol"]["search"] = dict(nsample = getvalue(self.lines,"nsample",int,False,default=10),
                                                seed = getvalue(self.lines,"seed",int,False,default=7),
                                                mo_rot_range= getvalue(self.lines,"mo_rot_range",float,False,default=numpy.pi)
                                               )

        elif self["jobcontrol"]["guess"] == "fromfile":
            self["jobcontrol"]["read_dir"] = getlist(self.lines,"read_dir",str,True)
        
        elif self["jobcontrol"]["guess"] == "fromorca":
            self["jobcontrol"]["orca_file"] = getvalue(self.lines,"orca_file",str,True)

        elif self["jobcontrol"]["guess"] == "ciguess":
            self["jobcontrol"]["ci_guess"] = getlist(self.lines,"ci_guess",int,True)

        elif self["jobcontrol"]["guess"] == "evlin":
            self["jobcontrol"]["read_dir"] = getlist(self.lines,"read_dir",str,True)
            self["jobcontrol"]["eigen_index"] = getvalue(self.lines,"eigen_index",int,default=+1)
            self["jobcontrol"]["linesearch_grid"] = getlist(self.lines,"linesearch_grid",int,False,default=[-numpy.pi,numpy.pi,51])
            self["jobcontrol"]["linesearch_nopt"] = getvalue(self.lines,"linesearch_nopt",int,False,default=5)
        elif self["jobcontrol"]["guess"] == "core":
            pass
        else:
            errstr = "'"+self["jobcontrol"]["guess"]+"' is not a valid option for keyword 'guess'"
            raise ValueError(errstr)

        if self["jobcontrol"]["noci"]:
            self["jobcontrol"]["noci_job"] = dict(lindep_tol = getvalue(self.lines,"lindep_tol",float,False,default=1e-8), 
                                                  plev = getvalue(self.lines,"plevel",int,False,default=1)
                                                 )
        if self["jobcontrol"]["analyse"]:
            self["jobcontrol"]["analyse"] = dict(states = getlist(self.lines,"states",str,False,default=["all"]),
                                                 orbital_plots = getlist(self.lines, "orbital_plots",int,True)
                                                )
        
        # Job control for computing oscillator strengths
        if self["jobcontrol"]["oscillator_strength"]:
            self["jobcontrol"]["oscillator_job"] = dict(ref_ind = getvalue(self.lines,"oscillator_reference",int,default=1)-1)

