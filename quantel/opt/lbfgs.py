#!/usr/bin/python3

import datetime
import numpy as np

def LBFGS(obj, thresh=1e-8, maxit=1000, index=None, plev=0):
    ''' This function is the one that we will run the Newton-Raphson calculation for a given NR_CASSCF object '''
    kernel_start_time = datetime.datetime.now() # Save initial time

    if plev>0: print()
    if plev>0: print("  Initializing L-BFGS solver...")
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

    # dimensions
    dim = obj.dim
    m = maxit
    
    # updates
    sk  = np.zeros((dim,m),np.float64)
    yk  = np.zeros((dim,m),np.float64)
    rho = np.zeros((m),np.float64)
    alpha = np.zeros(m,np.float64)

    if plev>0: print("  ==============================================")
    if plev>0: print("       {:^16s}    {:^8s}    {:^8s}".format("   Energy / Eh","Step Len","Error"))
    if plev>0: print("  ==============================================")
    k = 0
    while conv > thresh and k < maxit:
        # Get gradient
        g    = obj.get_gradient()
        conv = np.linalg.norm(g) * np.sqrt(1.0/g.size)

        # Take steepest descent on first step
        if k == 0:
            step    = - g

        else:
            yk[:,k-1]  = g - gref
            rho[k-1]   = (1.0 / np.dot(yk[:,k-1], sk[:,k-1]))

            if k > m: lim = k-m
            else:     lim = 0

            q    = g.copy()
            for i in range(k-1,lim-1,-1):
                alpha[i] = rho[i] * np.dot(sk[:,i], q)
                q -= alpha[i] * yk[:,i]

            gamma_k = np.dot(sk[:,k-1], yk[:,k-1]) / np.dot(yk[:,k-1], yk[:,k-1])
            step = gamma_k * q

            for i in range(lim,k):
                beta_i = rho[i] * np.dot(yk[:,i], step)
                step   = step + sk[:,i] * (alpha[i] - beta_i)
            step = - step

        gref = g.copy()
        sk[:,k] = step

#        # Rescale step length
        step_length = np.sqrt(np.dot(step,step))
        scale = min(step_length, r_trust) / step_length
        #scale = damp
        #step *= scale
        step_length = np.sqrt(np.dot(step,step))

        # energy change model
        dE_model = np.dot(step, g) * (1.0 - 0.5 * scale)
        obj.take_step(step)


 #       conv   = np.linalg.norm(g) * np.sqrt(1.0/g.size)
        e_ref  = energy 
        energy = obj.energy

        if(k > 0):   
            dE = energy - e_ref
            #print("Actual change, model = {: 10.6f} {: 10.6f} {: 10.6f}  {: 10.6f}".format( energy - e_ref, dE_model, dE / dE_model, r_trust))
            # scale trust radius
            if(dE / dE_model > tol_low and dE / dE_model < tol_high):
                r_trust *= scale_up
            else: 
                r_trust /= scale_up
            r_trust = min(max(r_trust, min_step),max_step)

        k += 1

        if plev>0: print(" {: 5d} {: 16.10f}    {:8.2e}    {:8.2e}".format(k,energy,step_length,conv))
    if plev>0: print("  ==============================================")
    kernel_end_time = datetime.datetime.now() # Save end time
    computation_time = kernel_end_time - kernel_start_time
    if plev>0: print("  L-BFGS walltime: ", computation_time.total_seconds(), " seconds")

    return
