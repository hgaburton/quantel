#!/usr/bin/env python

import sys, re, numpy, glob
from pyscf import gto
from pyscf.tools import cubegen

def analyse(ints, config):
    """Analyse either all the states or a given state"""
    print()
    print("---------------------------------------------------------------")
    print(" Analysing optimised solution(s)                               ")
    print("---------------------------------------------------------------")

    # Get information about the wavefunction
    wfnconfig = config["wavefunction"][config["wavefunction"]["method"]]
    if config["wavefunction"]["method"] == "esmf":
        from quantel.wfn.esmf import ESMF as WFN
    elif config["wavefunction"]["method"] == "casscf":
        from quantel.wfn.ss_casscf import SS_CASSCF as WFN
    elif config["wavefunction"]["method"] == "csf":
        from quantel.wfn.csf import GenealogicalCSF as WFN
    elif config["wavefunction"]["method"] == "rhf":
        from quantel.wfn.rhf import RHF as WFN
    else:
        raise ValueError("Wavefunction method not recognised")

    # Get list of states to be analysed
    fnames = []
    if config["jobcontrol"]["analyse"]["states"][0] == "all":
        for prefix in config["jobcontrol"]["read_dir"]:
            for i in range(len(glob.glob(prefix+"*.mo_coeff"))):
                fnames.append("{:s}{:04d}".format(prefix, i+1))
    else:
        fnames = config["jobcontrol"]["analyse"]["states"]

    for fname in fnames:
        print(" + Analysing ", fname)
        # Initialise optimisation object
        try: del myfun
        except: pass
        myfun = WFN(ints, **wfnconfig)
        myfun.read_from_disk(fname)
        # Store dipole and quadrupole
        #dip  = myfun.dipole
        
        #quit()
        #quad = myfun.quadrupole

        # Plot orbitals
        myfun.canonicalize()
        
        if(config["jobcontrol"]["integrals"]=='pyscf'):
            orbrange=config["jobcontrol"]["analyse"]["orbital_plots"]
            print(orbrange)
            print(myfun.nocc)
            print(myfun.nmo)
            print()
            print(myfun.nocc+orbrange[0], myfun.nocc+orbrange[1])

            if(len(orbrange)>0):
                for i in range(myfun.nocc+orbrange[0], myfun.nocc+orbrange[1]+1):
                    cubegen.orbital(ints.mol, fname+'.mo.{:d}.cube'.format(i+1), myfun.mo_coeff[:,i])
        else:
            ints.molden_orbs(myfun.mo_coeff,myfun.mo_occ,myfun.mo_energy)

        continue
        with open(fname+".analyse",'w+') as outF:
            outF.write("  Energy = {: 16.10f}\n".format(myfun.energy))
            outF.write("   <S^2> = {: 16.10f}\n".format(myfun.s2))
            #outF.write("   Index = {: 5d}\n".format(myfun.hess_index[0]))

            outF.write("\n  ----------------------------------------\n")
            outF.write("  Dipole moment:\n")
            outF.write("  ----------------------------------------\n")
            for (ix, x) in [(0,'x'),(1,'y'),(2,'z')]:
                outF.write("     {:5s}  {: 10.6f}\n".format(x, dip[ix]))

            outF.write("\n  ----------------------------------------\n")
            #outF.write("  Quadrupole moment:\n")
            #outF.write("  ----------------------------------------\n")
            #for (ix, x) in [(0,'x'),(1,'y'),(2,'z')]:
            #  for (iy, y) in [(0,'x'),(1,'y'),(2,'z')]:
            #    if iy<ix: continue
            #    outF.write("     {:5s}  {: 10.6f}\n".format(x+y, quad[ix,iy]))

            #outF.write("\n  ----------------------------------------\n")
            #outF.write("  Spatial moments:\n")
            #outF.write("  ----------------------------------------\n")
            #for (ix, x) in [(0,'x'),(1,'y'),(2,'z')]:
            #  for (iy, y) in [(0,'x'),(1,'y'),(2,'z')]:
            #    if iy<ix: continue
            #    outF.write("     {:5s}  {: 10.6f}\n".format(x+y, myfun.quad_nuc[ix,iy] - quad[ix,iy]))

            outF.write("\n  ----------------------------------------\n")
            outF.write("  Natural Orbitals:\n")
            outF.write("  ----------------------------------------\n")
            for i in range(myfun.nmo):
                outF.write(" {:5d}  {: 10.6f}  {: 10.6f}\n".format(i+1, myfun.mo_occ[i], myfun.mo_energy[i]))
            outF.write("  ----------------------------------------\n")

            #outF.write("\n  ----------------------------------------\n")
            #outF.write("  CI vector:\n")
            #outF.write("  ----------------------------------------\n")
            #for i in range(myfun.nDet):
            #    outF.write(" {:5d}  {: 10.6f}\n".format(i+1, myfun.mat_ci[i,0]))
            #outF.write("  ----------------------------------------\n")

            #outF.write("\n  ----------------------------------------\n")
            #outF.write("  Hessian eigenvalue structure:\n")
            #outF.write("     n      Eval       \n") 
            #outF.write("  ----------------------------------------\n")
            #for i, eigval in enumerate(eigval):
            #    outF.write(" {:5d}  {: 10.6f}\n".format(i, eigval))
            #outF.write("  ----------------------------------------\n")

             
            #ci_eig = 0.5 * numpy.linalg.eigvalsh(hess[nrot:,nrot:]) + myfun.energy
            #outF.write("\n  ----------------------------------------\n")
            #outF.write("  CI eigenvalues:\n")
            #outF.write("     n      Eval\n")
            #outF.write("  ----------------------------------------\n")
            #for i in range(ci_eig.size+1):
            #    if(i==0): outF.write(" {:5d}  {: 10.6f}\n".format(i, myfun.energy))
            #    else: outF.write(" {:5d}  {: 10.6f}\n".format(i, ci_eig[i-1]))
            #outF.write("  ----------------------------------------\n")

