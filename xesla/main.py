#!/usr/bin/python3

"""Main script for running state-specific energy landscape code

   Authors: 
       Hugh G. A. Burton  2020-
       Antoine Marie      2021
       Nick Lee           2022-
"""

import sys, argparse, numpy, time
from datetime import datetime, timedelta
from xesla.io.config import Config
from xesla.drivers import random_search, ci_guess, from_file
from pyscf import gto

def write_splash():
    print("===================================================")
    print("                      XESLA                        ")
    print("===================================================")
    print("  A library for exploring excited-state solutions  ")
    print("  in electronic structure theory.                  ")
    print("                                                   ")
    print("  Written by                                       ")
    print("     Antoine Marie, Nick Lee,                      ")
    print("     and Hugh G. A. Burton                         ")
    print("===================================================")


def main():
    # Parse in configuration file nuclear geometry
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", action="store", help="Input file containing calculation configuration")
    parser.add_argument("molecule_file", action="store", help="Text file containing molecular structure")
    args = parser.parse_args()

    # Startup
    write_splash()
    print(datetime.now().strftime("Today: %d %b %Y at %H:%M:%S"))
    start_time = time.monotonic()

    # Read the input file
    config = Config(args.input_file)
    config["molecule"]["atom"] = str(args.molecule_file)
    config.print()

    # Setup PySCF molecule
    mol = gto.Mole(**config["molecule"])
    mol.build()

    # Generate wavefunctions 
    wfnlist = None
    if config["jobcontrol"]["guess"] == "fromfile":
        wfnlist = from_file(mol, config)
    elif config["jobcontrol"]["guess"] == "random":
        wfnlist = random_search(mol, config)
    elif config["jobcontrol"]["guess"] == "ciguess":
        wfnlist = ci_guess(mol, config)
    else:
        errstr = "No wavefunctions have been defined"
        raise ValueError(errstr)


    if config["jobcontrol"]["ovlp_mat"]:
        # Compute the overlap matrix between solutions
        nstate = len(wfnlist)
        dist_mat = numpy.zeros((nstate,nstate))
        for i, state_i in enumerate(wfnlist):
            for j, state_j in enumerate(wfnlist):
                if(i<j): continue
                dist_mat[i,j] = state_i.overlap(state_j) 
                dist_mat[j,i] = dist_mat[i,j]

        numpy.savetxt('wfn_ov', dist_mat, fmt="% 8.6f")


    # Clean up
    end_time = time.monotonic()
    print("===================================================")
    print(" Calculation complete. Thank you for using XESLA!  ")
    print(" Total time = {:5.3f}s".format(timedelta(seconds=(end_time - start_time)).total_seconds()))
    print("===================================================")
