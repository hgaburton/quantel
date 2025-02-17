#!/usr/bin/python3

"""Main script for running state-specific energy landscape code

   Authors: 
       Hugh G. A. Burton  2020-
       Antoine Marie      2021
       Nicholas Lee       2022-
"""
# Set environment variables before importing numpy
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["OMP_NUM_THREADS"] = "1"
import argparse, numpy, time
from datetime import datetime, timedelta
from quantel import Molecule, LibintInterface
from quantel.io.config import Config
from quantel.drivers import random_search, from_file, from_orca, ci_guess, standard_guess, ev_linesearch, noci, overlap, analyse
from cProfile import Profile
from pstats import SortKey, Stats

def write_splash():
    print("====================================================")
    print("                      Quantel                       ")
    print("====================================================")
    print("  A library for exploring excited-state solutions   ")
    print("  in electronic structure theory.                   ")
    print("                                                    ")
    print("  Written by                                        ")
    print("     Antoine Marie, Nicholas Lee,                   ")
    print("     and Hugh G. A. Burton                          ")
    print("====================================================")


def main():
    # Parse in configuration file nuclear geometry
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", action="store", help="Input file containing calculation configuration")
    parser.add_argument("molecule_file", action="store", help="Text file containing molecular structure")
    args = parser.parse_args()

    # Make numpy print nicely
    numpy.set_printoptions(linewidth=100,precision=8,suppress=True,floatmode="fixed",edgeitems=10)

    # Startup
    write_splash()
    print(datetime.now().strftime("Today: %d %b %Y at %H:%M:%S"))
    start_time = time.monotonic()

    # Read the input file
    config = Config(args.input_file)
    config["molecule"]["atom"] = str(args.molecule_file)
    config.print()

    # Setup  molecule and integrals
    mol = Molecule(config["molecule"]["atom"], config["molecule"]["unit"])
    ints = LibintInterface(config["molecule"]["basis"],mol)
    # Generate wavefunctions 
    wfnlist = None
    if config["jobcontrol"]["guess"] == "fromfile":
        wfnlist = from_file(ints, config)
    elif config["jobcontrol"]["guess"] == "random":
        wfnlist = random_search(ints, config)
    elif config["jobcontrol"]["guess"] == "ciguess":
        wfnlist = ci_guess(ints, config)
    elif config["jobcontrol"]["guess"] == "fromorca":
        wfnlist = from_orca(ints, config)
    elif config["jobcontrol"]["guess"] == "evlin":
        wfnlist = ev_linesearch(ints, config)
    elif config["jobcontrol"]["guess"] == "standard":
        wfnlist = standard_guess(ints, config)
    else:
        errstr = "No wavefunctions have been defined"
        raise ValueError(errstr)

    #if config["jobcontrol"]["oscillator_strength"]:
    #    oscillator_strength(wfnlist, **config["jobcontrol"]["oscillator_job"])

    if config["jobcontrol"]["analyse"]:
        analyse(ints, config)
    if config["jobcontrol"]["noci"]:
        noci(wfnlist, **config["jobcontrol"]["noci_job"])
    elif config["jobcontrol"]["ovlp_mat"]:
        overlap(wfnlist)


    # Clean up
    end_time = time.monotonic()
    print()
    print("====================================================")
    print(" Calculation complete. Thank you for using Quantel! ")
    print(" Total time = {:5.3f}s".format(timedelta(seconds=(end_time - start_time)).total_seconds()))
    print("====================================================")
