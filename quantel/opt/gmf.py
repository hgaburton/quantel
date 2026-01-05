#!/usr/bin/python3

import datetime, sys
import numpy as np
from quantel.opt.davidson import Davidson

class GMF:
    '''
       Class to implement the generalised mode following optimisation for targeting saddle
       points of a particular Hessian index.

       This implementation follows the approach outlined in 
          Y. L. A. Schmerwitz, G. Levi, H. Jonsson
          J. Chem. Theory Cmput. 19, 3634 (2023)
    '''

    def __init__(self, **kwargs):
        '''Initialise the GMF instance'''
        self.control = dict()
        self.control["minstep"] = 0.01
        self.control["maxstep"] = 0.2
        self.control["max_subspace"] = 10
        self.control["with_transport"] = True
        self.control["with_canonical"] = True
        self.control["canonical_interval"] = 10

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
        maxstep = self.control["maxstep"]

        # Initialise reference energy
        eref = obj.energy
        dim = obj.dim
        grad = obj.gradient
        prec = obj.get_preconditioner()
        # Print here to see if this is the slow step
        # what is this doing then - what are the different Davidson values taking in 
        eigval, evec = Davidson(nreset=50).run(obj.approx_hess_on_vec,prec,index,maxit=300,tol=1e-4,plev=1)
        gmod, evec = self.get_gmf_gradient(obj,grad,index,eigval,evec)

        if plev>0:
            print(f"    > Num. MOs     = {obj.nmo: 6d}")
            print(f"    > Num. params  = {obj.dim: 6d}")
            print(f"    > Max subspace = {max_subspace: 6d}")
            print(f"    > Max step     = {maxstep: 6.3f}")
            print()

        # Initialise lists for subspace vectors
        v_step = []
        v_gmod = [gmod]

        if plev>0: print("  ================================================================")
        if plev>0: print("       {:^16s}    {:^8s}    {:^8s}".format("   Energy / Eh","Step Len","Error"))
        if plev>0: print("  ================================================================")

        converged = False
        qn_count = 0
        for istep in range(maxit+1):
            # Get gradient and check convergence
            #conv = np.linalg.norm(grad) * np.sqrt(1.0/grad.size)
            conv = np.linalg.norm(grad,ord=np.inf) #stationary point check 
            ecur = obj.energy

            if istep > 0 and plev > 0:
                print(" {: 5d} {: 16.10f}    {:8.2e}    {:8.2e}    {:10s}".format(
                      istep, ecur, step_length, conv, comment))
            elif plev > 0:
                print(" {: 5d} {: 16.10f}                {:8.2e}".format(istep, ecur, conv))
            sys.stdout.flush()

            # Check if we have convergence
            if(conv < thresh):
                converged = True
                break

            # Get L-BFGS quasi-Newton step
            step = self.get_lbfgs_step(v_gmod,v_step)
            comment = ""
            qn_count += 1
            if(np.dot(step,grad) > 0):
                # Need to make sure  s.g < 0 to maintain positive-definite L-BFGS Hessian 
                print("Step has positive overlap with gradient - reversing direction")
                step *= -1
                comment = comment + "reversed "

            # Truncate the max step size
            lstep = np.linalg.norm(step)
            if(lstep > self.control["maxstep"]):
                step = self.control["maxstep"] * step / lstep
                comment = "truncated"
            else: 
                comment = ""
  
            # Check for step length converged
            step_length = np.linalg.norm(step)
            if(step_length < thresh*thresh):
                return True

            # Take the step
            obj.take_step(step)
            # Save step
            v_step.append(step.copy())

            # Pseudo-canonicalize orbitals if requested
            X = None
            if(self.control["with_canonical"] and np.mod(qn_count,self.control["canonical_interval"])==0):
                #print("  Pseudo-canonicalising the orbitals")
                X = obj.canonicalize()

            # Parallel transport previous vectors
            if(self.control["with_transport"]):
                v_gmod = [obj.transform_vector(v, step, X) for v in v_gmod] 
                v_step = [obj.transform_vector(v, step, X) for v in v_step] 
                for i in range(index):
                    evec[:,i] = obj.transform_vector(evec[:,i], step, X)

            # Compute n lowest eigenvalues
            prec = obj.get_preconditioner()
            eigval, evec = Davidson(nreset=50).run(obj.approx_hess_on_vec,prec,index,xguess=evec,tol=1e-4,plev=0)

            # Compute new GMF gradient (need to parallel transport Hessian eigenvector)
            grad = obj.gradient
            gmod, evec = self.get_gmf_gradient(obj,grad,index,eigval,evec)
            v_gmod.append(gmod.copy())

            # Remove oldest vectors if subspace is saturated
            if(len(v_step)>max_subspace):
                v_gmod.pop(0)
                v_step.pop(0)

        if plev>0: print("  ================================================================")

        # Save end time and report duration
        kernel_end_time = datetime.datetime.now()
        computation_time = kernel_end_time - kernel_start_time
        if plev>0: print("  Generalised mode following walltime: ", computation_time.total_seconds(), " seconds")

        obj.get_davidson_hessian_index(guess=evec)


        return converged

    def get_gmf_gradient(self,obj,grad,n,e,x):
        """ Compute modified gradient for n-index saddle point search using generalised mode following

            This gradient corresponds to Eq. (11) in  
              Y. L. A. Schmerwitz, G. Levi, H. Jonsson
              J. Chem. Theory Cmput. 19, 3634 (2023)
        """
        if(n==0):
            return grad, None

        # Compute n lowest eigenvalues
        #prec = obj.get_preconditioner()
        #e, x = Davidson(nreset=50).run(obj.approx_hess_on_vec,prec,n,xguess=xguess,tol=1e-4,plev=0)

        # Then project gradient as required
        if(e[n-1] < 0):
            gmod = grad - 2 * x @ (x.T @ grad)
        else:
            gmod = np.zeros((grad.size))
            for i in range(n):
                if(e[i] >= 0):
                    gmod = gmod - x[:,i] * np.dot(x[:,i], grad)

        return gmod, x
    

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
        r = gamma_k * q
        for i in range(nvec):
            beta = rho[i] * np.dot(yk[i], r)
            r = r + sk[i] * (alpha[i] - beta) 
        return -r
