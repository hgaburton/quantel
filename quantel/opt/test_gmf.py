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
        #self.control["maxstep"] = 0.2
        self.control["maxstep"] = 0.15
        self.control["max_subspace"] = 10 #Size of the maximum Krylov subspace
        self.control["with_transport"] = True #Parallel transport of previous iterations
        self.control["with_canonical"] = True #canonical? why?
        self.control["canonical_interval"] = 10 #?
        
        # this just makes sure that key word arguments, kwargs were filled in correctly
        for key in kwargs:
            if not key in self.control.keys():
                print("ERROR: Keyword [{:s}] not recognised".format(key))
            else: 
                self.control[key] = kwargs[key]

    def run(self, obj, thresh=1e-8, maxit=200, index=0, plev=1):
        ''' Run the optimisation for a particular objective function obj.
            
            obj must have the following methods implemented:
              + energy
              + gradient
              + dim
              + take_step()
              + transform_vector()
              + get_preconditioner() this is the positive definite diagonal approximation to the Hessian
              + canonicalize()
        '''
        # Save initial time
        kernel_start_time = datetime.datetime.now()

        if plev>0: print()
        if plev>0: print( "  Initializing Generalized Mode Following...")
        if plev>0 and (not index == None): print(f"    Target Hessian index = {index: 5d}") 

        # Extract key parameters
        max_subspace = self.control["max_subspace"]
        maxstep = self.control["maxstep"]
        dim = obj.dim  

        if plev>0:
            print(f"    > Num. MOs     = {obj.nmo: 6d}")
            print(f"    > Num. params  = {obj.dim: 6d}")
            print(f"    > Max subspace = {max_subspace: 6d}")
            print(f"    > Max step     = {maxstep: 6.3f}")
            print()

        # Initialise lists for subspace vectors
        v_step = []
        v_gmod = []

        if plev>0: print("  ================================================================")
        if plev>0: print("       {:^16s}    {:^8s}    {:^8s}".format("   Energy / Eh","Step Len","Error"))
        if plev>0: print("  ================================================================")

        converged = False
        qn_count = 0
            

        # This loop does our GDM/LBFGS routine with just a modified gradient step 
        for istep in range(maxit+1):

            ecur = obj.energy
            grad = obj.gradient
            conv = np.linalg.norm(grad,ord=np.inf)
            
            if istep > 0 and plev > 0:
                print(" {: 5d} {: 16.10f}    {:8.2e}    {:8.2e}    {:10s}".format(
                      istep, ecur, np.linalg.norm(step), conv, comment))
            sys.stdout.flush()

            # Check if we have convergence
            if(conv < thresh):
                converged = True
                break

            # Check if line search conditions are satisfied 
            #################################################
            # Cant do a normal wolfe line search as we dont know to what objective function the modified gradient corresponds to.
            ###################################################
            # Pseudo-canonicalize orbitals if requested: we should always do this!
            X = None
            if(self.control["with_canonical"] and np.mod(qn_count,self.control["canonical_interval"])==0):
                X = obj.canonicalize()

            # Parallel transport previous vectors
            guess = None
            #Should included a reset run here?? 
            #if istep!=0 and not reset and (self.control["with_transport"]):
            if istep!=0 and (self.control["with_transport"]):
                v_gmod = [obj.transform_vector(v, step, X) for v in v_gmod] 
                v_step = [obj.transform_vector(v, step, X) for v in v_step]
                guess = np.zeros(evec.shape) 
                for i in range(index):
                    guess[:,i] = obj.transform_vector(evec[:,i], step, X) 
            # Compute n lowest eigenvalues
            prec = obj.get_preconditioner()
            eigval, evec = Davidson(nreset=50).run(obj.approx_hess_on_vec,prec,index,xguess=guess,tol=1e-4,plev=0, maxit=500)   

            # Compute new GMF gradient
            ecur = obj.energy
            grad = obj.gradient
            gmod = self.get_gmf_gradient(obj,grad,index,eigval,evec)
            # Get L-BFGS quasi-Newton step
            v_gmod.append(gmod.copy())

            step = self.get_lbfgs_step(v_gmod,v_step,prec)
            #step = self.new_get_lbfgs_step(v_gmod,v_step)#doing this in the modified gradient picture, for this we dont know what the analytic Hessian will look like which is why we cant do the preconditioning step in the get_lbfgs_step? but we take absolute value of the diagonals anyways so why not try?
            
            comment = ""
            qn_count += 1
            # Truncate the max step size, rescale the step
            lstep = np.linalg.norm(step)
            if(lstep > self.control["maxstep"]):
                step = self.control["maxstep"] * step / lstep
                comment = comment + " truncated"

            if(np.dot(step,gmod) > 0):
                # check step.gmod < 0, i.e. we have maintained positive-definite L-BFGS Hessian
                # is this actually helping us though, since we want a postive definite LBFGS Hessian which guarantees we are going downhill on our modified surface?
                #print("Step has positive overlap with gradient - reversing direction")
                
                step *= -1
                comment = comment + " reversed"

            # Take the step
            obj.take_step(step)
            # Save step
            v_step.append(step.copy())

            
            # Remove oldest vectors if subspace is saturated
            if(len(v_gmod)>max_subspace):
                v_gmod.pop(0)
                v_step.pop(0)

        if plev>0: print(" ================================================================")

        # Save end time and report duration
        kernel_end_time = datetime.datetime.now()
        computation_time = kernel_end_time - kernel_start_time
        if plev>0: 
            print("  Generalised mode following walltime: ", computation_time.total_seconds(), " seconds")
            print("Check convergence: ", converged)
            sys.stdout.flush()
        return converged

    def get_gmf_gradient(self,obj,grad,index,eigval,evec):
        """No changes made with respect to the base gmf"""
        if(index==0):
            return grad

        # Project gradient as required
        if(eigval[index-1] < 0):
            gmod = grad - 2 * evec @ (evec.T @ grad)
        else:
            gmod = np.zeros((grad.size))
            for i in range(index):
                if(eigval[i] >= 0):
                # selects out the positive or zero modes and then take the opposite of the projection onto this mode
                    gmod = gmod - evec[:,i] * np.dot(evec[:,i], grad)
        return gmod 
    
    def get_lbfgs_step(self,v_grad,v_step,prec):
        """ No changes made with respect to the base gmf"""
        # Subspace size
        nvec = len(v_step)
        assert(len(v_grad)==nvec+1) #right since we should have updated with current gradient

        # Clip the preconditioner to avoid numerical issues
        thresh=0.1
        prec = np.sqrt(np.clip(prec,thresh,None)) # prec is just the same as diag
        # This step sets all values less than thresh equal to thresh and then square roots. Square rooting to get ready to convert from pseudocanonical orbital to EWCs 

        
        # Get sk, yk, and rho in energy weighted coordinates
        # These are already parallel transported (if choosing to transport)
        sk = [v_step[i] * prec for i in range(nvec)]
        yk = [(v_grad[i+1] - v_grad[i]) / prec for i in range(nvec)]
        rho = [1.0 / np.dot(yk[i], sk[i]) for i in range(nvec)]

        # Get gamma_k
        # Nocedal's formula to compute the H0k if one of the Hessian eigenvalues is negative: H0k = gamma_k * unit_matrix 
        gamma_k = np.dot(sk[-1], yk[-1]) / np.dot(yk[-1], yk[-1]) if (nvec > 0) else 1 
        
        # Initialise step from last gradient
        q = v_grad[-1].copy() / prec #this is the hessian update formula, and since we use a diagonal approximation to the Hessian this is fine! 

        # Compute alpha and beta terms
        # what are the alpha and beta terms? two loop recursion scheme  
        alpha = np.empty(nvec)
        for i in range(nvec-1,-1,-1):
            alpha[i] = rho[i] * np.dot(sk[i], q) 
            q = q - alpha[i] * yk[i]

        # Apply preconditioner, which can be unity or scaled unity matrix 
        r = q * gamma_k

        # Second loop of L-BFGS
        for i in range(nvec):
            beta = rho[i] * np.dot(yk[i], r)
            r = r + sk[i] * (alpha[i] - beta) 

        # Convert step back to non-energy weighted coordinates
        # alpha_k p_k = p_k = -H_k del f_k = - r/prec  
        return - r / prec
