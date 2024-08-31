#!/usr/bin/python3

import datetime, sys
import numpy as np
from quantel.utils.linalg import orthogonalise  
import scipy

class LBFGS:
    '''
       Class to implement quasi-Newton L-BFGS algorithm for optimising minima.

       This implementation follows the approach outlined in 
          Numerical Optimization, J. Nocedal and S. J. Wright
       with a backtracking routine to avoid overstepping.
    '''

    def __init__(self, **kwargs):
        '''Initialise the GMF instance'''
        self.control = dict()
        self.control["minstep"] = 0.01
        self.control["maxstep"] = 0.2
        self.control["max_subspace"] = 20
        self.control["backtrack_scale"] = 0.1

        for key in kwargs:
            if not key in self.control.keys():
                print("ERROR: Keyword [{:s}] not recognised".format(key))
            else: 
                self.control[key] = kwargs[key]

    def run(self, obj, thresh=1e-8, maxit=100, index=0, plev=1):
        ''' Run the optimisation for a particular objective function obj.
            
            obj must have the following methods implemented:
              + energy
              + gradient
              + dim
              + take_step()
              + transform_vector()
              + get_preconditioner()
              + canonicalize()
        '''
        # Save initial time
        kernel_start_time = datetime.datetime.now()

        # Canonicalise, might put Fock matrices in more diagonal form
        obj.canonicalize()

        if plev>0: print()
        if plev>0: print( "  Initializing Generalized Mode Following...")
        if plev>0 and (not index == None): print(f"    Target Hessian index = {index: 5d}") 

        # Extract key parameters
        max_subspace = self.control["max_subspace"]
        rescale = self.control["backtrack_scale"]

        # Initialise reference energy
        eref = obj.energy
        dim = obj.dim
        grad = obj.gradient

        if plev>0:
            print(f"    > Num. MOs     = {obj.nmo: 6d}")
            print(f"    > Num. params  = {dim: 6d}")
            print(f"    > Max subspace = {max_subspace: 6d}")
            print(f"    > Backtracking = {rescale: 6.3f}")
            print()

        # Initialise lists for subspace vectors
        v_step = []
        v_grad = [grad]

        if plev>0: print("  ================================================================")
        if plev>0: print("       {:^16s}    {:^8s}    {:^8s}".format("   Energy / Eh","Step Len","Error"))
        if plev>0: print("  ================================================================")

        converged = False
        for istep in range(maxit+1):
            # Get gradient and check convergence
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

            # Get L-BFGS quasi-Newton step
            qn_step = self.get_lbfgs_step(v_grad,v_step)
            comment = ""
            if(np.dot(qn_step,grad) > 0):
                # Need to make sure  s.g < 0 to maintain positive-definite L-BFGS Hessian 
                print("Step has positive overlap with gradient - reversing direction")
                qn_step *= -1
                comment = comment + "reversed "

            # Apply backtracking to satisfy Wolfe sufficient decrease
            # Eq. (3.6a) in Nocedal and Wright (page 34)
            sg = np.dot(qn_step,grad)
            alpha = 1
            obj.save_last_step()
            backtrack = False
            while not backtrack:
                obj.take_step(alpha * qn_step)
                if obj.energy <= eref + 1e-4 * alpha * sg: 
                    backtrack = True
                else:
                    alpha *= rescale
                    comment = comment + "backtrack "
                obj.restore_last_step()
            step = alpha * qn_step

            # Truncate the max step size
            lstep = np.linalg.norm(step)
            if(lstep > self.control["maxstep"]):
                step = self.control["maxstep"] * step / lstep
                comment = comment + "truncated "
  
            # Check for step length converged
            step_length = np.linalg.norm(step)
            if(step_length < thresh*thresh):
                return True

            # Take the step
            obj.take_step(step)

            # Save step
            v_step.append(step.copy())

            # Parallel transport previous vectors
            v_grad = [obj.transform_vector(v, 0.5 * step) for v in v_grad] 
            v_step = [obj.transform_vector(v, 0.5 * step) for v in v_step] 

            # Save new gradient
            grad = obj.gradient
            v_grad.append(grad.copy())

            # Remove oldest vectors if subspace is saturated
            if(len(v_step)>max_subspace):
                v_grad.pop(0)
                v_step.pop(0)

            # Increment the iteration counter
            istep += 1

        if plev>0: print("  ================================================================")

        # Save end time and report duration
        kernel_end_time = datetime.datetime.now()
        computation_time = kernel_end_time - kernel_start_time
        if plev>0: print("  L-BFGS walltime: ", computation_time.total_seconds(), " seconds")

        return converged


    def get_lbfgs_step(self,v_grad,v_step):
        """ Compute the L-BFGS step from previous gradient and step vectors

            This routine follows Algorithm 7.4 on page 178 in 
               Numerical Optimization, J. Nocedal and S. J. Wright
        """
        # Subspace size
        nvec = len(v_step)
        assert(len(v_grad)==nvec+1)

        # Get sk, yk, and rho
        sk = v_step
        yk = [v_grad[i+1] - v_grad[i] for i in range(nvec)]
        rho = [1.0 / np.dot(yk[i], sk[i]) for i in range(nvec)]

        # Get gamma_k
        gamma_k = np.dot(sk[-1], yk[-1]) / np.dot(yk[-1], yk[-1]) if (nvec > 0) else 1 

        # Initialise step from last gradient
        q = v_grad[-1].copy()

        # Compute alpha and beta terms
        alpha = np.empty(nvec)
        for i in range(nvec-1,-1,-1):
            alpha[i] = rho[i] * np.dot(sk[i], q) 
            q = q - alpha[i] * yk[i]
        # Including gamma_k is vital for preconditioning the step length
        r = gamma_k * q
        for i in range(nvec):
            beta = rho[i] * np.dot(yk[i], r)
            r = r + sk[i] * (alpha[i] - beta) 
        return - r
