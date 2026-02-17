#!/usr/bin/python3

import numpy, glob

def ev_linesearch(ints, config):
    """Perform zero-eigenvector linesearch"""

    print("-----------------------------------------------")
    print(" Reading solutions from file")
    print("    + Wavefunction:       {:s}".format(config["wavefunction"]["method"]))
    print("-----------------------------------------------")

    # Get information about the wavefunction
    wfnconfig = config["wavefunction"][config["wavefunction"]["method"]]
    if config["wavefunction"]["method"] == "esmf":
        from quantel.wfn.esmf import ESMF as WFN
        ref_ci = numpy.identity(WFN(ints, **wfnconfig).ndet)
    elif config["wavefunction"]["method"] == "casscf":
        from quantel.wfn.ss_casscf import SS_CASSCF as WFN
        ref_ci = numpy.identity(WFN(ints, **wfnconfig).ndet)
    elif config["wavefunction"]["method"] == "csf":
        from quantel.wfn.csf import CSF as WFN
    elif config["wavefunction"]["method"] == "rhf":
        from quantel.wfn.rhf import RHF as WFN
    elif config["wavefunction"]["method"] == "roks":
        from quantel.wfn.roks import ROKS as WFN
    else:
        raise ValueError("Wavefunction method not recognised")

    # Select the optimiser
    optconfig = config["optimiser"][config["optimiser"]["algorithm"]]
    if config["optimiser"]["algorithm"] == "eigenvector_following":
        from quantel.opt.eigenvector_following import EigenFollow as OPT
    elif config["optimiser"]["algorithm"] == "mode_control":
        from quantel.opt.mode_controlling import ModeControl as OPT
    else:
        raise ValueError("Optimiser algorithm not recognised")

    # Initialise wavefunction list
    wfn_list = []
    e_list   = []
    i_list   = []

    # Reconverge target solutions
    target_index = config["optimiser"]["keywords"]["index"]
    count = 0
    for prefix in config["jobcontrol"]["read_dir"]:
        print(" Reading solutions from directory {:s}".format(prefix))
        #reading solutions so presumably this would be reading in from a previous geometry, would have solutions that have 
        # been organised into directories with convergence to come from above or below - i would rather that we had this automatically.
        # Need to count the number of states to converge
        nstates = len(glob.glob(prefix+"*.mo_coeff"))
        for i in range(nstates):
            old_tag = "{:s}{:04d}".format(prefix, i+1)

            # Initialise optimisation object
            try: del myfun
            except: pass
            myfun = WFN(ints, **wfnconfig)
            myfun.read_from_disk(old_tag)

            # Run the optimisation
            myopt = OPT(**optconfig)
            if not myopt.run(myfun, **config["optimiser"]["keywords"]):
                continue

            # Get Hessian indices
            hindices = myfun.get_hessian_index()
            if (hindices[0] != target_index) and (target_index is not None):
                continue

            # Compare solution against previously found states
            new = True
            for prev, otherwfn in enumerate(wfn_list):
                if 1.0 - abs(myfun.overlap(otherwfn)) < config["jobcontrol"]["dist_thresh"]:
                    new = False
                    break

            # Save the solution if it is a new one!
            if new: 
                if config["wavefunction"]["method"] == "esmf":
                    myfun.canonicalize()
                # Get the prefix for this solution
                count += 1
                tag = "{:04d}".format(count)

                # Save the object to disck
                myfun.save_to_disk(tag)

                # Save energy and indices
                e_list.append(myfun.energy)
                i_list.append(hindices[0])

                # Deallocate integrals to reduce memory footprint
                myfun.deallocate()
                wfn_list.append(myfun.copy())
            else: 
                print("  Solution matches previous solution...",prev+1)

        # Attempt to reconverge states
        nsearch = len(wfn_list)
        for isol in range(nsearch):
            # Initialise optimisation object
            try: del myfun
            except: pass
            myfun = wfn_list[isol].copy(integrals=True)
            
            # Get the Hessian
            hess = myfun.hessian
            # Diagonalise hessian
            eigval, eigvec = numpy.linalg.eigh(hess)

            # Identify target hessian index
            eigen_target = config["jobcontrol"]["eigen_index"]
            #so the target hessian index - i guess is going to be either +1 or -1 ?
            if eigen_target > 0: # so eigen_target = 1 
                indzero = numpy.argmin(numpy.abs(eigval)+1e10*(eigval<0)) + eigen_target - 1
                #indzero this is trying to find where the near zero index - but the near zero one from above
            else:
                indzero = numpy.argmin(numpy.abs(eigval)+1e10*(eigval>0)) + eigen_target + 1
                #indzero this is trying to find where the near zero index - from below

            print()
            print(" Performing eigenvector linesearch for solution {:5d}:".format(isol+1))
            print("  Search along eigenvector {:5d} with eigenvalue = {: 16.10f}".format(indzero+1, eigval[indzero]))

            # Get step and energy function
            # gets the step direction from indzero
            step = eigvec[:,indzero]
            myfun.save_last_step()
            #see how the energy varies along that step direction
            def get_energy(x):
                myfun.take_step(x * step)
                energy = myfun.energy
                myfun.restore_last_step()
                return energy
            
            # Compute linesearch values 
            # alpha defines the kind of grid we are looking at
            ls = numpy.array([[alpha, get_energy(alpha)] for alpha in numpy.linspace(*config["jobcontrol"]["linesearch_grid"])])
            # so that ls = [step len , energy]
 
            # Compute numerical gradients at each point
            ### compute via finite differences the absolute value of the gradient at each point
            gls = numpy.zeros((ls.shape[0]-1,ls.shape[1]))
            gls[:,0] = 0.5 * (ls[:-1,0] + ls[1:,0]) #this is saving the average value betwen two adjacent step lengths?  
            gls[:,1] = numpy.abs((ls[1:,1] - ls[:-1,1]) / (ls[1:,0] - ls[:-1,0])) #finite differences abs gradient at each of these midpoints

            # Test new stationary points
            nopt = config["jobcontrol"]["linesearch_nopt"]
            for ind in numpy.argsort(gls[:,1])[:nopt]:
                #choosing the npot number of points with the smallest absolute value of the gradient as starting points for the optimsation
                x = gls[ind,0] # the midpoints where we have calculated numerical gradients 

                print("\n  Approximate stationary point at x = {: 16.10f}:".format(x))
                print("  -----------------------------------------------")
                newfun = myfun.copy(integrals=True)
                newfun.take_step(x * step)

                # Run the optimisation
                myopt = OPT(**optconfig)
                if not myopt.run(newfun, **config["optimiser"]["keywords"]):
                    continue

                # Compare solution against previously found states
                new = True
                for prev, otherwfn in enumerate(wfn_list):
                    if 1.0 - abs(newfun.overlap(otherwfn)) < config["jobcontrol"]["dist_thresh"]:
                        new = False
                        break

                # Save the solution if it is a new one!
                if new: 
                    hindices = newfun.get_hessian_index()
                    if config["wavefunction"]["method"] == "esmf":
                        newfun.canonicalize()
                    # Get the prefix for this solution
                    count += 1
                    tag = "{:04d}".format(count)

                    # Save the object to disck
                    newfun.save_to_disk(tag)

                    # Save energy and indices
                    e_list.append(newfun.energy)
                    i_list.append(hindices[0])

                    # Deallocate integrals to reduce memory footprint
                    newfun.deallocate()
                    wfn_list.append(newfun.copy())
                else: 
                    print("  Solution matches previous solution...",prev+1)


    numpy.savetxt('energy_list', numpy.array([e_list]),fmt="% 16.10f")
    numpy.savetxt('ind_list', numpy.array([i_list]),fmt="% 5d")

    print()
    print(" Read from file complete... Identified {:5d} unique solutions".format(len(wfn_list)))
    print("--------------------------------------------------------------")
    print()

    return wfn_list
