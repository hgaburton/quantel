#!/usr/bin/env python

import sys, re, numpy, glob
from pyscf.tools import cubegen

def analyse(mol, config):
    """Analyse either all the states or a given state"""

    print("---------------------------------------------------------------")
    print(" Analysing optimised solution(s)                               ")
    print("---------------------------------------------------------------")

    # Get information about the wavefunction
    wfnconfig = config["wavefunction"][config["wavefunction"]["method"]]
    if config["wavefunction"]["method"] == "esmf":
        from exelsis.wfn.esmf import ESMF as WFN
        ref_ci = numpy.identity(WFN(mol, **wfnconfig).nDet)
        ndet = ref_ci.shape[1]
    elif config["wavefunction"]["method"] == "casscf":
        from exelsis.wfn.ss_casscf import SS_CASSCF as WFN
        ref_ci = numpy.identity(WFN(mol, **wfnconfig).nDet)
        ndet = ref_ci.shape[1]
    elif config["wavefunction"]["method"] == "csf":
        from exelsis.wfn.csf import CSF as WFN
        ndet = 0

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
        myfun = WFN(mol, **wfnconfig)
        myfun.read_from_disk(fname)

        # Store dipole and quadrupole
        dip  = myfun.dipole
        quad = myfun.quadrupole

        # Plot orbitals
        orbrange=config["jobcontrol"]["analyse"]["orbital_plots"]
        if(len(orbrange)>0):
            for i in range(orbrange[0]-1, orbrange[1]):
                cubegen.orbital(mol, fname+'.mo.{:d}.cube'.format(i+1), myfun.mo_coeff[:,i])

        myfun.canonicalise()
        # Plot orbitals
        if(len(orbrange)>0):
            for i in range(orbrange[0]-1, orbrange[1]):
                cubegen.orbital(mol, fname+'.no.{:d}.cube'.format(i+1), myfun.mo_coeff[:,i])

        s2 = myfun.s2
        hindices = myfun.get_hessian_index()

        hess = myfun.hessian
        eigval, eigvec = numpy.linalg.eigh(hess)
        nrot = myfun.nrot 

        with open(fname+".analyse",'w+') as outF:
            outF.write("  Energy = {: 16.10f}\n".format(myfun.energy))
            outF.write("   <S^2> = {: 16.10f}\n".format(myfun.s2))
            outF.write("   Index = {: 5d}\n".format(numpy.sum(eigval<0)))

            outF.write("\n  ----------------------------------------\n")
            outF.write("  Dipole moment:\n")
            outF.write("  ----------------------------------------\n")
            for (ix, x) in [(0,'x'),(1,'y'),(2,'z')]:
                outF.write("     {:5s}  {: 10.6f}\n".format(x, dip[ix]))

            outF.write("\n  ----------------------------------------\n")
            outF.write("  Quadrupole moment:\n")
            outF.write("  ----------------------------------------\n")
            for (ix, x) in [(0,'x'),(1,'y'),(2,'z')]:
              for (iy, y) in [(0,'x'),(1,'y'),(2,'z')]:
                if iy<ix: continue
                outF.write("     {:5s}  {: 10.6f}\n".format(x+y, quad[ix,iy]))

            outF.write("\n  ----------------------------------------\n")
            outF.write("  Spatial moments:\n")
            outF.write("  ----------------------------------------\n")
            for (ix, x) in [(0,'x'),(1,'y'),(2,'z')]:
              for (iy, y) in [(0,'x'),(1,'y'),(2,'z')]:
                if iy<ix: continue
                outF.write("     {:5s}  {: 10.6f}\n".format(x+y, myfun.quad_nuc[ix,iy] - quad[ix,iy]))

            outF.write("\n  ----------------------------------------\n")
            outF.write("  Natural Orbitals:\n")
            outF.write("  ----------------------------------------\n")
            for i in range(myfun.norb):
                outF.write(" {:5d}  {: 10.6f}  {: 10.6f}\n".format(i+1, myfun.mo_occ[i], myfun.mo_energy[i]))
            outF.write("  ----------------------------------------\n")

            outF.write("\n  ----------------------------------------\n")
            outF.write("  CI vector:\n")
            outF.write("  ----------------------------------------\n")
            for i in range(myfun.nDet):
                outF.write(" {:5d}  {: 10.6f}\n".format(i+1, myfun.mat_ci[i,0]))
            outF.write("  ----------------------------------------\n")

            outF.write("\n  ----------------------------------------\n")
            outF.write("  Hessian eigenvalue structure:\n")
            outF.write("     n      Eval       |V_scf|    |V_ci| \n") 
            outF.write("  ----------------------------------------\n")
            for i, eigval in enumerate(eigval):
                vec_scf = eigvec[:nrot,i]
                vec_ci  = eigvec[nrot:,i]
                outF.write(" {:5d}  {: 10.6f}  {: 10.6f} {: 10.6f}\n".format(
                      i, eigval, numpy.linalg.norm(vec_scf), numpy.linalg.norm(vec_ci)))
            outF.write("  ----------------------------------------\n")

             
            ci_eig = 0.5 * numpy.linalg.eigvalsh(hess[nrot:,nrot:]) + myfun.energy
            outF.write("\n  ----------------------------------------\n")
            outF.write("  CI eigenvalues:\n")
            outF.write("     n      Eval\n")
            outF.write("  ----------------------------------------\n")
            for i in range(ci_eig.size+1):
                if(i==0): outF.write(" {:5d}  {: 10.6f}\n".format(i, myfun.energy))
                else: outF.write(" {:5d}  {: 10.6f}\n".format(i, ci_eig[i-1]))
            outF.write("  ----------------------------------------\n")

