#!/usr/bin/python3

import re, numpy

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

    def __getvalue(self, target, typ, required=False, default=None):
        """Get the value of a keyword with a single argument"""
        for line in self.lines:
            if re.match(target, line) is not None:
                return typ(re.split(r'\s+', line.strip())[-1])
        if required:
            errstr = "Keyword '"+target+"' was not found"
            raise ValueError(errstr)
        elif default is not None:
            return default

    def __getlist(self, target, typ, required=False):
        """Get the value of a keyword with a list of arguments"""
        for line in self.lines:
            if re.match(target, line) is not None:
                return [typ(x) for x in re.split(r'\s+', line.strip())[1:]]
        if required:
            errstr = "Keyword '"+target+"' was not found"
            raise ValueError(errstr)
        return []

    def __getbool(self, target, required=False, default=None):
        """Get the value for a boolean keyword"""
        for line in self.lines:
            if re.match(target, line) is not None:
                value = str(re.split(r'\s+', line.strip())[-1])
                if value in ["1","True","true"]:
                    self["molecule"][target] = True
                elif value in ["0","False","false"]:
                    self["molecule"][target] = False
                else:
                    errstr = "Boolean '"+target+"' keyword value '"+value+"' is not valid" 
                    raise ValueError(errstr)
        if required:
            errstr = "Keyword '"+target+"' was not found"
            raise ValueError(errstr)
        elif default is not None:
            return default


    def parse_molecule(self):
        """Read keywords that define the molecular system"""
        self["molecule"] = dict(charge = self.__getvalue("charge",int,True),
                                spin = self.__getvalue("spin",int,True),
                                basis = self.__getvalue("basis",str,True), 
                                unit = self.__getvalue("units",str,False,"Ang")
                               )

            
    def parse_wavefunction(self):
        """Read keywords that define the wavefunction model"""
        self["wavefunction"] = dict(method = self.__getvalue("method",str,True).lower())

        # Get keywords for each allowed method
        if self["wavefunction"]["method"] == "esmf":
            # TODO: Add support for triplet states 
            self["wavefunction"]["esmf"] = dict(ref_allowed = self.__getbool("with_ref",False,True))

        elif self["wavefunction"]["method"] == "casscf":
            self["wavefunction"]["casscf"] = dict(active_space = self.__getlist("active_space",int,True))

        elif self["wavefunction"]["method"] == "csf":
            self["wavefunction"]["csf"] = dict(g_coupling = self.__getvalue("genealogical_coupling",str,True),
                                               mo_basis = self.__getvalue("mo_basis",str,True),
                                               active = self.__getlist("active_orbitals",int,True),
                                               active_space=self.__getlist("active_space",int,True),
                                               core = self.__getlist("core_orbitals",int,True),
                                               permutation = self.__getlist("coupling_permutation",int,True),
                                               stot = self.__getvalue("total_spin",int,True)
                                              )


    def parse_optimiser(self):
        """Read keywords that define the optimiser we use"""
        self["optimiser"] = dict(algorithm = self.__getvalue("algorithm",str,False,default="eigenvector_following").lower())

        if self["optimiser"]["algorithm"] == "eigenvector_following":
            self["optimiser"]["eigenvector_following"] = dict(minstep = self.__getvalue("minstep",float,False,default=0),
                                                              rtrust  = self.__getvalue("rstrust",float,False,default=0.15),
                                                              maxstep = self.__getvalue("maxstep",float,False,default=numpy.pi),
                                                              hesstol = self.__getvalue("hesstol",float,False,1e-16)
                                                             )
        elif self["optimiser"]["algorithm"] == "mode_control":
            self["optimiser"]["mode_control"] = dict(minstep = self.__getvalue("minstep",float,False,default=0),
                                                    rtrust  = self.__getvalue("rstrust",float,False,default=0.15),
                                                    maxstep = self.__getvalue("maxstep",float,False,default=numpy.pi),
                                                    hesstol = self.__getvalue("hesstol",float,False,1e-16)
                                                    )

        else:
            errstr = "Requested optimiser '"+self["optimiser"]["algorithm"]+"' is not available"
            raise ValueError(errstr)


        self["optimiser"]["keywords"] = dict(thresh = self.__getvalue("convergence",float,False,default=1e-8),
                                             maxit = self.__getvalue("maxit",int,False,default=500),
                                             index = self.__getvalue("target_index",int,False,default=None)
                                            )

        

    def parse_jobcontrol(self):
        """Parse keywords that define how jobs are run"""
        self["jobcontrol"] = dict(guess = self.__getvalue("guess",str,False,default="random").lower(), 
                                  noci  = self.__getbool("noci",False,default=False),
                                  dist_thresh = self.__getvalue("dist_tresh",float,False,default=1e-8)
                                  ) 
        
        if self["jobcontrol"]["guess"] == "random":
            self["jobcontrol"]["search"] = dict(nsample = self.__getvalue("nsample",int,False,default=10),
                                                seed = self.__getvalue("seed",int,False,default=7)
                                               )

        elif self["jobcontrol"]["guess"] == "fromfile":
            self["jobcontrol"]["read_dir"] = self.__getlist("read_dir",str,True)

        elif self["jobcontrol"]["guess"] == "ciguess":
            self["jobcontrol"]["ci_guess"] = self.__getlist("ci_guess",int,True)
