#!/usr/bin/python3

import datetime, sys
import numpy as np
from .trust_radius import TrustRadius

class ModeControl:

    def __init__(self, **kwargs):
        '''Initialise the Newton-Raphson procedure'''
        self.control = dict()
        self.control["hesstol"] = 1e-16
        self.control["minstep"] = 0.0
        self.control["maxstep"] = 0.2
        self.control["rtrust"]  = 0.01
        self.control["hesstol"] = 1e-16

        for key in kwargs:
            if not key in self.control.keys():
                print("ERROR: Keyword [{:s}] not recognised".format(key))
            else: 
                self.control[key] = kwargs[key]

        # Initialise the trust radius controller
        self.__trust = TrustRadius(self.control["rtrust"], self.control["minstep"], self.control["maxstep"])

    def run(self, obj, thresh=1e-8, maxit=100, index=None, plev=1):
        ''' This function runs the optimisation'''

        kernel_start_time = datetime.datetime.now() # Save initial time

        if plev>0: print()
        if plev>0: print("  Initializing mode-controlled Newton-Raphson solver...")

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
                      istep, eref, cur_hind, step_length, conv, comment))
            elif plev > 0:
                print(" {: 5d} {: 16.10f}    {:8.0f}                {:8.2e}".format(istep, eref, cur_hind, conv))
            sys.stdout.flush()

            if(index == None):
                index = np.sum(hess_eig < 0)

            # Check if we have convergence
            if(conv < thresh): 
                converged = True
                break

            # Get the trial step
            step, dE_model, comment = self.get_step(grad, hess_vec, hess_eig)

            # Transform step back into full space and take step
            step_length = np.linalg.norm(step)
            if(step_length < thresh*thresh):
                return True
            obj.take_step(step)

            # Get actual energy change
            dE = obj.energy - eref

            # Assess trust radius
            if istep > 0:
                # Save reference energy if we accept step, otherwise undo the step
                if self.__trust.accept_step(dE, dE_model, step_length):
                    eref = obj.energy
                else:
                    # Otherwise undo the step
                    comment = "No step"
                    obj.restore_last_step()

            # Increment the iteration counter
            istep += 1

        if plev>0: print("  ================================================================")
        kernel_end_time = datetime.datetime.now() # Save end time
        computation_time = kernel_end_time - kernel_start_time
        if plev>0: print("  Eigenvector-following walltime: ", computation_time.total_seconds(), " seconds")

        return converged


    def get_step(self, grad, hess_vec, hess_eig):
        '''Compute the optimal eigenvector-following step and the analogous steepest-descent step'''

        # Transform gradient into Hessian eigenbasis
        gt = hess_vec.T.dot(grad)

        # Compute step in Hessian eigenbasis 
        qn_t = np.zeros(gt.shape) # NR-style step
 
        # Get steps with given number of uphill directions
        upcount = 0
        for i in range(hess_eig.size):
            if(abs(hess_eig[i]) < self.control["hesstol"]): 
                print("   Zero Hessian index: {: 16.10e}".format(hess_eig[i]))
                continue
            qn_t[i] = - gt[i] / hess_eig[i]

        # Truncate step using Trust radius
        step_t, comment = self.__trust.truncated_step(qn_t)
        
        # Compute trust radius model energy change
        dE_model = np.dot(gt, step_t) + 0.5 * np.einsum('i,i,i', hess_eig, step_t, step_t)

        # Transform step back into full space and return
        step = hess_vec.dot(step_t)

        return step, dE_model, comment
