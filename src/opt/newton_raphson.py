#!/usr/bin/python3

import datetime
import numpy as np

def NewtonRaphson(obj, thresh=1e-8, maxit=100, index=None, plev=0):
    ''' This function is the one that we will run the Newton-Raphson calculation for a given NR_CASSCF object '''
    kernel_start_time = datetime.datetime.now() # Save initial time

    if plev>0: print()
    if plev>0: print("  Initializing Newton-Raphson solver...")
    istep = 0
    conv = 1
    energy = 1e10
    damp = 0.1

    # Trust radius
    min_step = 0.01
    max_step = 2 * np.pi
    tol_high = 1.2
    tol_low  = 0.8
    scale_up = 1.5
    r_trust  = 1.0

    if plev>0: print("  ==============================================")
    if plev>0: print("       {:^16s}    {:^8s}    {:^8s}".format("   Energy / Eh","Step Len","Error"))
    if plev>0: print("  ==============================================")
    while conv > thresh and istep < maxit:
        # Get gradient and Hessian
        g = obj.get_gradient()
        H = obj.get_hessian()

        # Compute Newton-Raphson step
        
        eig, vec = np.linalg.eigh(H) 
        step     = np.zeros(g.shape)
        for i in range(eig.size):
            if(abs(eig[i]) < 1e-10): 
                continue
            if index == None:
                step -= (np.dot(vec[:,i],g) / eig[i]) * vec[:,i]
            else:
                if i < index:
                    step -= (- np.dot(vec[:,i],g) / np.abs(eig[i])) * vec[:,i]
                else:
                    step -= (  np.dot(vec[:,i],g) / np.abs(eig[i])) * vec[:,i]

        # Rescale step length
        step_length = np.sqrt(np.dot(step,step))
        scale = min(step_length, r_trust) / step_length
        #scale = damp
        step *= scale
        step_length = np.sqrt(np.dot(step,step))

        # energy change model
        dE_model = np.dot(step, g) * (1 - 0.5 * scale)
        obj.take_step(step)

        if np.max(np.abs(g)) < conv and damp < 4:
            damp = damp * 1.1
        elif np.max(np.abs(g)) > conv and damp > 0.01:
            damp = damp * 0.8

        conv   = np.linalg.norm(g) * np.sqrt(1.0/g.size)
        e_ref  = energy 
        energy = obj.energy

        if(istep > 0):   
            dE = energy - e_ref
            #print("Actual change, model = {: 10.6f} {: 10.6f} {: 10.6f} ".format( energy - e_ref, dE_model, dE / dE_model))
            # scale trust radius
            if(dE / dE_model > tol_low and dE / dE_model < tol_high):
                r_trust *= scale_up
            else: 
                r_trust /= scale_up
            r_trust = min(max(r_trust, min_step),max_step)

        istep += 1

        if plev>0: print(" {: 5d} {: 16.10f}    {:8.2e}    {:8.2e}".format(istep,energy,step_length,conv))
    if plev>0: print("  ==============================================")
    kernel_end_time = datetime.datetime.now() # Save end time
    computation_time = kernel_end_time - kernel_start_time
    if plev>0: print("  Newton-Raphson walltime: ", computation_time.total_seconds(), " seconds")

    return
