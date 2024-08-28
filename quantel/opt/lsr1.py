#!/usr/bin/python3

import datetime, sys
import numpy as np
from .trust_radius import TrustRadius
from quantel.utils.linalg import orthogonalise  
import scipy

class SR1:

    def __init__(self, **kwargs):
        '''Initialise the eigenvector following instance'''
        self.control = dict()
        self.control["minstep"] = 0.01
        self.control["maxstep"] = np.pi
        self.control["rtrust"]  = 0.15
        self.control["hesstol"] = 1e-16
        self.control["precmin"] = 1e0

        for key in kwargs:
            if not key in self.control.keys():
                print("ERROR: Keyword [{:s}] not recognised".format(key))
            else: 
                self.control[key] = kwargs[key]

        # Initialise the trust radius controller
        self.__trust = TrustRadius(self.control["rtrust"], self.control["minstep"], self.control["maxstep"])

    def run(self, obj, thresh=1e-8, maxit=100, index=0, plev=1, max_subspace=10):
        ''' This function is the one that we will run the Newton-Raphson calculation for a given NR_CASSCF object '''
        kernel_start_time = datetime.datetime.now() # Save initial time

        # Canonicalise, might put Fock matrices in more diagonal form
        obj.canonicalize()

        if plev>0: print()
        if plev>0: print( "  Initializing L-SR1 optimisation...")

        # Initialise reference energy
        eref = obj.energy
        dim = obj.dim
        grad = obj.gradient

        # Get parameters
        precmin = self.control["precmin"]

        if plev>0:
            print(f"    > Num. MOs     = {obj.nmo: 6d}")
            print(f"    > Num. params  = {obj.dim: 6d}")
            print(f"    > precmin      = {precmin: 6.2e}")
            print()

        # Initialise lists for subspace vectors
        v_step = []
        v_grad = [grad.copy()]

        if plev>0: print("  ================================================================")
        if plev>0: print("       {:^16s}    {:^8s}    {:^8s}".format("   Energy / Eh","Step Len","Error"))
        if plev>0: print("  ================================================================")

        converged = False
        for istep in range(maxit+1):
            # Get gradient and check convergence
            grad = obj.gradient
            conv = np.linalg.norm(grad) * np.sqrt(1.0/grad.size)
            eref = obj.energy

            if istep > 0 and plev > 0:
                print(" {: 5d} {: 16.10f}    {:8.2e}    {:8.2e}    {:10s}".format(
                      istep, eref, step_length, conv, comment))
            elif plev > 0:
                print(" {: 5d} {: 16.10f}                {:8.2e}".format(istep, eref, conv))
            sys.stdout.flush()

            # Check if we have convergence
            if(conv < thresh): 
                converged = True
                break

            # Get preconditioner
            prec = obj.get_preconditioner()
            for i in range(dim):
                if(abs(prec[i]) < precmin):
                    prec[i] = np.sign(prec[i]) * precmin

            # Get L-SR1 quasi-Newton step
            qn_step = self.get_lsr1_step(v_grad,v_step,prec,grad)

            # Apply step truncation
            step, comment = self.__trust.truncated_step_norm(qn_step)

            # Compute model energy change
            alpha = np.dot(step, qn_step) / np.dot(qn_step, qn_step)
            dE_model = (1 - 0.5 * alpha) * np.dot(step, grad)
  
            # Check for step length converged
            step_length = np.linalg.norm(step)
            if(step_length < thresh*thresh):
                return True

            # Take the step
            obj.take_step(step)

            # Save step
            v_step.append(step.copy())
            # Get new gradient
            gnew = obj.gradient

            # Get actual energy change
            dE = obj.energy - eref

            # Assess trust radius
            if istep > 0:
                # Save reference energy if we accept step, otherwise undo the step
                if self.__trust.accept_step(dE, dE_model, step_length):
                    eref = obj.energy
                    # Save the new origin 
                    obj.save_last_step()
                    # Parallel transport vectors
                    v_grad = [obj.transform_vector(v, 0.5 * step) for v in v_grad] 
                    v_step = [obj.transform_vector(v, 0.5 * step) for v in v_step] 
                else:
                    # Otherwise undo the step
                    comment = "No step"
                    # Parallel transport gradient back to origin
                    gnew = obj.transform_vector(gnew, -0.5 * step)
                    # Restore step
                    obj.restore_last_step()

            # Save new gradient
            v_grad.append(gnew)

            # Remove oldest vectors if subspace is saturated
            if(len(v_step)>max_subspace):
                v_grad.pop(0)
                v_step.pop(0)

            # Increment the iteration counter
            istep += 1

        if plev>0: print("  ================================================================")
        if(converged):
            print("Outcome = {:6d} {: 16.10f} {:6.4e} {:6d}".format(np.sum(np.linalg.eigvalsh(obj.hessian)<0), 
                                   obj.energy, np.linalg.norm(obj.gradient), istep))
        else:
            print("Outcome = failed")
        kernel_end_time = datetime.datetime.now() # Save end time
        computation_time = kernel_end_time - kernel_start_time
        if plev>0: print("  Eigenvector-following walltime: ", computation_time.total_seconds(), " seconds")

        return converged

    def get_lsr1_step(self,v_grad,v_step,prec,grad,tol=1e-8):
        """Compute the SR1 Hessian approximation in the subspace"""
        # Number of subspace vectors
        nvec = len(v_step)

        # Initialise approx inverse Hessian from preconditioner
        B = np.diag(np.power(prec,-1))

        # Compute the inverse Hessian
        s = v_step
        y = [v_grad[i+1] - v_grad[i] for i in range(nvec)]
        for k in range(nvec):
            rk = s[k] - B @ y[k]
            rho = np.dot(y[k],rk)
            if(abs(rho) >= tol * np.linalg.norm(y[k]) * np.linalg.norm(rk)):
                B = B + np.outer(rk, rk) / rho
                
        # Compute QN step and return
        return - B @ grad
