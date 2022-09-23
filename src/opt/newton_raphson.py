#!/usr/bin/python3

import datetime, sys
import numpy as np

class NewtonRaphson:

    def __init__(self, **kwargs):
        '''Initialise the Newton-Raphson procedure'''
        self.control = dict()
        self.control["hesstol"] = 1e-16

    def run(self, obj, thresh=1e-8, maxit=100, index=None, plev=1):
        ''' This function runs the optimisation'''

        kernel_start_time = datetime.datetime.now() # Save initial time

        if plev>0: print()
        if plev>0: print("  Initializing Newton-Raphson solver...")

        # Initialise reference energy
        eref = obj.energy

        if plev>0: print("  ================================================================")
        if plev>0: print("       {:^16s}    {:^8s}    {:^8s}    {:^8s}".format("   Energy / Eh","Index","Step Len","Error"))
        if plev>0: print("  ================================================================")

        converged = False
        for istep in range(maxit+1):
            # Get gradient and check convergence
            grad = obj.gradient
            conv = np.linalg.norm(grad) * np.sqrt(1.0/grad.size)
            eref = obj.energy

            # Get Hessian eigen-decomposition
            hess_eig, hess_vec = np.linalg.eigh(obj.hessian) 
            cur_hind = np.sum(hess_eig<0)

            if istep > 0 and plev > 0:
                print(" {: 5d} {: 16.10f}    {:8.0f}    {:8.2e}    {:8.2e}    {:10s}".format(
                      istep, eref, cur_hind, step_length, conv, ""))
            elif plev > 0:
                print(" {: 5d} {: 16.10f}    {:8.0f}                {:8.2e}".format(istep, eref, cur_hind, conv))
            sys.stdout.flush()

            if(index == None):
                index = np.sum(hess_eig < 0)

            # Check if we have convergence
            if(conv < thresh): 
                converged = True
                break

            # Compute Newton-Raphson step
            step     = np.zeros(grad.shape)
            for i, eigval in enumerate(hess_eig):
                if(abs(eigval) < self.control['hesstol']): 
                    continue
                else:
                    step -= (np.dot(hess_vec[:,i],grad) / eigval) * hess_vec[:,i]
            print(hess_eig)
            print(step)

            # Transform step back into full space and take step
            step_length = np.linalg.norm(step)
            if(step_length < thresh*thresh):
                return converged
            obj.take_step(step)

            # Increment the iteration counter
            istep += 1

        if plev>0: print("  ================================================================")
        kernel_end_time = datetime.datetime.now() # Save end time
        computation_time = kernel_end_time - kernel_start_time
        if plev>0: print("  Eigenvector-following walltime: ", computation_time.total_seconds(), " seconds")

        return converged
