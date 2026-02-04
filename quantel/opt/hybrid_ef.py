#!/usr/bin/python3

import datetime, sys
import numpy as np

from quantel.opt.lbfgs import LBFGS
from quantel.opt.davidson import Davidson

class HybridEF:
    def __init__(self, **kwargs):
        '''Initialise the Hybrid eigenvector-following instance'''

        self.control = dict()
        self.control["maxstep"] = 0.5

        for key in kwargs:
            if not key in self.control.keys():
                print("ERROR: Keyword [{:s}] not recognised".format(key))
            else: 
                self.control[key] = kwargs[key]

    def run(self, obj, thresh=1e-6, maxit=100, index=0, plev=1, approx_hess=False):
        ''' Run the optimisation for a particular objective function obj.
            
            obj must have the following methods implemented:
              + energy
              + gradient
              + dim
              + take_step()
              + transform_vector()
              + hess_on_vec() or approx_hess_on_vec()
              + canonicalize()
        '''
        kernel_start_time = datetime.datetime.now() # Save initial time
        if plev>0: print()
        if plev>0: print( "  Initializing Hybrid Eigenvector-Following...")

        dim = obj.dim
        if(dim == 0): return True

        if plev>0:
            print(f"    > Num. params          = {dim: 6d}")
            print(f"    > Max uphill step      = {self.control['maxstep']: 6.3f}")
            print(f"    > Target Hessian index = {index: 5d}")
            print(f"    > Approx. hess_on_vec  = {approx_hess}")

        if plev>0: print("  ================================================================")
        if plev>0: print("       {:^16s}    {:^8s}    {:^8s}    {:^8s}".format("   Energy / Eh","Index","Step Len","Error"))
        if plev>0: print("  ================================================================")

        # Canonicalise to start, speeds up Davidson
        obj.canonicalize()

        if(approx_hess):
            if(not hasattr(obj, "approx_hess_on_vec")):
                raise RuntimeError("Objective function does not have approx_hess_on_vec() method implemented")
            hv = obj.approx_hess_on_vec
        else:
            if(not hasattr(obj, "hess_on_vec")):
                raise RuntimeError("Objective function does not have hess_on_vec() method implemented")
            hv = obj.hess_on_vec
        davidson = Davidson(nreset=50,basis_per_root=8)

        evec=None
        converged = False
        for istep in range(maxit+1):
            # Get gradient and check convergence
            grad = obj.gradient
            eref = obj.energy
            conv = np.linalg.norm(grad,ord=np.inf)

            # Get lowest Hessian eigenvectors using Davidson
            prec = obj.get_preconditioner(abs=False)
            eigval, evec = davidson.run(hv,prec,index+1,tol=1e-4,plev=plev-1,xguess=evec)
            cur_ind = np.sum(eigval<0)
            st_cur_ind = "{:2s}{:<d}".format('>=' if cur_ind > index else ' ', cur_ind)

            
            # Check if we have convergence
            if(conv < thresh): 
                if(cur_ind == index):
                    # We have the correct index, so we return converged
                    converged = True
                    comment = "Converged"
                else:
                    # We have converged to the wrong index, so take a small step and continue
                    if(cur_ind > index):
                        step = 0.1 * evec[:,cur_ind-1]
                        comment = "Index too high, stepped along highest known -ve mode"
                    elif(cur_ind < index):
                        step = 0.1 * evec[:,cur_ind]
                        comment = "Index too low, stepped along lowest known +ve mode"            
                    # Take the escaping step
                    step_length = np.linalg.norm(step)
                    obj.take_step(step)
            else:
                # Build step uphill
                step = np.zeros(obj.dim)
                for ivec in range(index):
                    vi, ei = evec[:,ivec], np.abs(eigval[ivec])
                    F = grad.dot(vi)
                    h = 2*F / (ei * (1.0 + np.sqrt(1.0 + 4*F*F/(ei*ei))))
                    step += h*vi
                
                # Max step size for uphill component
                if(np.linalg.norm(step) > self.control["maxstep"]):
                    step = step * (self.control["maxstep"] / np.linalg.norm(step))
                    comment = "Truncated"
                else: 
                    comment = ""
                step_length = np.linalg.norm(step)

                # Take the uphill step
                obj.take_step(step)

                # Optimise energy in directions orthogonal to uphill steps
                LBFGS(with_transport=False,with_canonical=False).run(obj,proj_vec=evec[:,:index],maxit=10,plev=plev-1)

            # Report our progress
            if not converged:
                print(" {: 5d} {: 16.10f}    {:^8s}    {:8.2e}    {:8.2e}    {:10s}".format(
                      istep, eref, st_cur_ind, step_length, conv, comment))
            else:
                print(" {: 5d} {: 16.10f}    {:^8s}                {:8.2e}    {:10s}".format(
                    istep, eref, st_cur_ind, conv, comment))
                break
            sys.stdout.flush()

        if plev>0: print("  ================================================================")
        kernel_end_time = datetime.datetime.now() # Save end time
        computation_time = kernel_end_time - kernel_start_time
        if plev>0: print("  Hybrid eigenvector-following walltime: ",computation_time.total_seconds(), " seconds")
        sys.stdout.flush()

        return converged