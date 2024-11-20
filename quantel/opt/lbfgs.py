#!/usr/bin/python3

import datetime, sys
import numpy as np

class LBFGS:
    '''
       Class to implement quasi-Newton L-BFGS algorithm for optimising minima.

       This implementation follows the approach outlined in 
          Numerical Optimization, J. Nocedal and S. J. Wright
       with a backtracking routine to avoid overstepping.
    '''
    def __init__(self, **kwargs):
        '''Initialise the LBFGS instance'''
        self.control = dict()
        self.control["minstep"] = 0.01
        self.control["maxstep"] = 0.5
        self.control["max_subspace"] = 20
        self.control["backtrack_scale"] = 0.1
        self.control["with_transport"] = True
        self.control["with_canonical"] = True
        self.control["canonical_interval"] = 10
        self.control['gamma_preconditioner'] = False

        for key in kwargs:
            if not key in self.control.keys():
                print("ERROR: Keyword [{:s}] not recognised".format(key))
            else: 
                self.control[key] = kwargs[key]

    def run(self, obj, thresh=1e-6, maxit=100, plev=1, ethresh=1e-8, index=0):
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

        if plev>0: print()
        if plev>0: print( "  Initializing L-BFGS optimisation...")

        # Extract key parameters
        max_subspace = self.control["max_subspace"]
        dim = obj.dim

        if plev>0:
            print(f"    > Num. MOs       = {obj.nmo: 6d}")
            print(f"    > Num. params    = {dim: 6d}")
            print(f"    > Max subspace   = {max_subspace: 6d}")
            print(f"    > Backtracking   = {self.control['backtrack_scale']: 6.3f}")
            print(f"    > Parallel tr.   = {self.control['with_transport']}")
            print(f"    > Pseudo-canon.  = {self.control['with_canonical']}")
            print(f"    > Canon interval = {self.control['canonical_interval']}")
            print(f"    > Hybrid prec.   = {not self.control['gamma_preconditioner']}")
            print()

        # Initialise lists for subspace vectors
        v_step = []
        v_grad = []

        if plev>0: print("  ================================================================")
        if plev>0: print("       {:^16s}    {:^8s}    {:^8s}".format("   Energy / Eh","Step Len","Max(|G_i|)"))
        if plev>0: print("  ================================================================")

        zero_step = np.zeros(obj.dim)
        converged = False
        n_rescale = 0
        qn_count = 0
        reset = False
        for istep in range(maxit+1):
            # Get energy, gradient and check convergence
            ecur = obj.energy
            grad = obj.gradient
            c1 = np.linalg.norm(grad,ord=2) * np.sqrt(1.0/dim)
            c2 = np.linalg.norm(grad,ord=np.inf)
            MaxGrad = np.linalg.norm(grad,ord=np.inf)
            conv = c2
            
            if istep > 0 and plev > 0:
                print(" {: 5d} {: 16.10f}    {:8.2e}     {:8.2e}    {:10s}".format(
                      istep, ecur, np.max(np.abs(step)), conv, comment))
                #print(f" {istep: 5d} {ecur: 16.10f}   {ecur-eref: 8.2e}   {RMSDP: 8.2e}   {MaxDP: 8.2e}   {MaxGrad: 8.2e}")
            elif plev > 0:
                print(" {: 5d} {: 16.10f}                 {:8.2e}".format(istep, ecur, conv))
            sys.stdout.flush()

            # Check if we have convergence
            if(conv < thresh):
                converged = True
                break


            # Check if we have sufficient decrease
            if(istep == 0 or reset):
                wolfe1 = True
                reset = False
            else:
                wolfe1 = (ecur - eref) <= 1e-4 * np.dot(step,gref)

            # Obtain new step
            comment = ""
            if(wolfe1):
                # Number of rescaled steps in a row
                n_rescale = 0
                # Accept the step, update origin and compute new L-BFGS step
                obj.save_last_step()

                # Pseudo-canonicalize orbitals if requested
                X = None
                if(self.control["with_canonical"] and np.mod(qn_count,self.control["canonical_interval"])==0):
                   #conv<1e-2):
                    X = obj.canonicalize()
                    grad = obj.gradient

                # Save reference energy and gradient
                eref = ecur
                gref = grad.copy()

                # Parallel transport previous vectors
                if(self.control["with_transport"]):
                    v_grad = [obj.transform_vector(v, step, X) for v in v_grad] 
                    v_step = [obj.transform_vector(v, step, X) for v in v_step] 
                elif(self.control["with_canonical"]):
                    v_grad = [obj.transform_vector(v, zero_step, X) for v in v_grad] 
                    v_step = [obj.transform_vector(v, zero_step, X) for v in v_step] 

                # Save new gradient
                v_grad.append(grad.copy())

                # Get L-BFGS quasi-Newton step
                prec = obj.get_preconditioner()
                step = self.get_lbfgs_step(v_grad,v_step,prec)
                qn_count += 1

                # Need to make sure s.g < 0 to maintain positive-definite L-BFGS Hessian 
                if(np.dot(step,gref) > 0):
                    print("  Step has positive overlap with gradient - reversing direction")
                    step *= -1
                    reset = True
                
                # Truncate the max step size
                #lscale = self.control["maxstep"] / np.linalg.norm(step)
                lscale = self.control["maxstep"] / np.max(np.abs(step))
                if(lscale < 1):
                    step = lscale * step
                    comment = "truncated"

                if(np.linalg.norm(step,ord=np.inf) < 1e-10):
                    print("  Step size indicates convergence")
                    converged = True
                    break

            else:
                print(f"  Insufficient decrease - rescaling step size by {self.control['backtrack_scale']:4.2f}")
                # Insufficient decrease in energy, restore last position and reset L-BFGS
                obj.restore_last_step()

                # Rescale step and remove last step from memory
                step = step * self.control["backtrack_scale"]
                v_step.pop(-1)
                comment = "backtrack"
                n_rescale = n_rescale + 1 
                reset = True

            # Save step and length
            step_length = np.linalg.norm(step)
            v_step.append(step.copy())

            # Save step statistics
            RMSDP = np.linalg.norm(step) / np.sqrt(dim)
            MaxDP = np.linalg.norm(step,ord=np.inf)

            # Take the step
            obj.take_step(step)

            # Remove oldest vectors if subspace is saturated
            if(len(v_step)>max_subspace):
                v_grad.pop(0)
                v_step.pop(0)

            if(reset):
                comment = "reset L-BFGS"
                v_grad = []
                v_step = []

            # Increment the iteration counter
            istep += 1

        if plev>0: print("  ================================================================")

        # Save end time and report duration
        kernel_end_time = datetime.datetime.now()
        computation_time = (kernel_end_time - kernel_start_time).total_seconds()
        if(not converged):
            if plev>0: print(f"  L-BFGS failed to converge in {istep: 6d} iterations ({computation_time: 6.2f} seconds)")
        else:
            if plev>0: print(f"  L-BFGS converged in {istep: 6d} iterations ({computation_time: 6.2f} seconds)")
        sys.stdout.flush()


        return converged


    def get_lbfgs_step(self,v_grad,v_step,prec):
        """ Compute the L-BFGS step from previous gradient and step vectors

            This routine follows Algorithm 7.4 on page 178 in 
               Numerical Optimization, J. Nocedal and S. J. Wright
            
            The step is preconditioned by the diagonal elements of an approximate Hessian.
            If any of the diagonal elements are negative, the approximate inverse Hessian is defined
            using Nocedal's formula.
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
        # Use a dynamic scaling for maximum preconditioner
        thresh=1
        prec = np.clip(prec,thresh,None)

        # Compute alpha and beta terms
        alpha = np.empty(nvec)
        for i in range(nvec-1,-1,-1):
            alpha[i] = rho[i] * np.dot(sk[i], q) 
            q = q - alpha[i] * yk[i]

        # Including preconditioning is vital to control the step length
        if(self.control["gamma_preconditioner"] or np.any(prec <= 0)):
            # If out preciditioner is not positive definite, we assume that we are not 
            # in a quadratic region and use Nocedal's formula to approximate the inverse Hessian
            r = gamma_k * q
        else:
            # Recall preconditioner is Hessian, not inverse
            r = q / prec

        # Second loop of L-BFGS
        for i in range(nvec):
            beta = rho[i] * np.dot(yk[i], r)
            r = r + sk[i] * (alpha[i] - beta) 

        return - r
