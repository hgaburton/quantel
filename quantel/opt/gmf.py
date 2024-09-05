#!/usr/bin/python3

import datetime, sys
import numpy as np
from .trust_radius import TrustRadius
from quantel.utils.linalg import orthogonalise  
import scipy

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
        self.control["rtrust"]  = 0.15
        self.control["max_subspace"] = 20

        for key in kwargs:
            if not key in self.control.keys():
                print("ERROR: Keyword [{:s}] not recognised".format(key))
            else: 
                self.control[key] = kwargs[key]

        # Initialise the trust radius controller
        self.__trust = TrustRadius(self.control["rtrust"], self.control["minstep"], self.control["maxstep"])

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
        gmod, evec = self.get_gmf_gradient(obj,grad,index)

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
            step = self.get_lbfgs_step(v_gmod,v_step)
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

            # Parallel transport previous vectors
            v_gmod = [obj.transform_vector(v, 0.5 * step) for v in v_gmod] 
            v_step = [obj.transform_vector(v, 0.5 * step) for v in v_step] 

            # Compute new GMF gradient (need to parallel transport Hessian eigenvector)
            xguess = np.empty((dim,index))
            for i in range(index):
                xguess[:,i] = obj.transform_vector(evec[:,i], 0.5 * step)
            # Save gradient
            grad = obj.gradient
            gmod, evec = self.get_gmf_gradient(obj,grad,index,xguess=evec)
            v_gmod.append(gmod.copy())

            # Remove oldest vectors if subspace is saturated
            if(len(v_step)>max_subspace):
                v_gmod.pop(0)
                v_step.pop(0)

            # Increment the iteration counter
            istep += 1

        if plev>0: print("  ================================================================")

        # Save end time and report duration
        kernel_end_time = datetime.datetime.now()
        computation_time = kernel_end_time - kernel_start_time
        if plev>0: print("  Generalised mode following walltime: ", computation_time.total_seconds(), " seconds")

        return converged


    def get_gmf_gradient(self,obj,grad,n,xguess=None):
        """ Compute modified gradient for n-index saddle point search using generalised mode following

            This gradient corresponds to Eq. (11) in  
              Y. L. A. Schmerwitz, G. Levi, H. Jonsson
              J. Chem. Theory Cmput. 19, 3634 (2023)
        """
        if(n==0):
            return grad, None

        # Compute n lowest eigenvalues
        e, x = self.get_Hessian_eigenpairs(obj,grad,n,xguess=xguess)

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


    def get_Hessian_eigenpairs(self,obj,g0,n,xguess=None,max_iter=100,eps=1e-5,tol=1e-2,nreset=10):
        """ Compute the lowest n eigenvectors and eigenvalues of the Hessian using the 
            Davidson algorithm. 

            The Hessian vector product H @ v is computed approximately using the forward finite 
            difference approach given by Eq. (12) in
              Y. L. A. Schmerwitz, G. Levi, H. Jonsson
              J. Chem. Theory Cmput. 19, 3634 (2023)

            An initial guess for the eigenvectors can be provided by the optional argument xguess.
        """
        # Initialise Krylov subspace
        dim = g0.size
        K   = np.empty((dim, 0))
        
        # Shifted objected
        shift = obj.copy()

        # Get approximate diagonal of Hessian
        Qdiag = obj.get_preconditioner()

        # If no guess provided, start with identity
        if(xguess is None):
            inds = np.argsort(Qdiag)[:n]
            K = 0.01 * np.ones((dim,n))
            for i,j in enumerate(inds):
                K[j,i] = 1
        else:
            assert(xguess.shape[1] == n)
            K = xguess.copy()
        K = orthogonalise(K,np.identity(dim),fill=False)

        # Initialise HK vectors
        HK = np.empty((dim, 0))

        # Loop over iterations
        for it in range(max_iter):
            # Form new HK vectors as required
            for ik in range(HK.shape[1],K.shape[1]):
                # Get step
                sk = K[:,ik]
                # Get forward gradient
                shift.initialise(obj.mo_coeff,spin_coupling=obj.spin_coupling,integrals=False)
                shift.take_step(eps * sk)
                g1 = shift.gradient.copy()
                # Parallel transport back to current position
                g1 = obj.transform_vector(g1, -0.5 * eps * sk)
                # Get approximation to H @ sk
                H_sk = (g1 - g0) / eps
                # Add to HK space
                HK = np.column_stack([HK,H_sk]) 

            # Solve Krylov subproblem
            A = K.T @ HK
            e, y = np.linalg.eigh(A)
            # Extract relevant eigenvalues (e) and eigenvectors (x)
            e = e[:n]
            y = y[:,:n]
            x = K @ y
            
            # Compute residuals
            r = HK @ y - x @ np.diag(e)
            residuals = np.max(np.abs(r),axis=0)
            # Check convergence
            if all(res < tol for res in residuals):
                break

            # Reset Krylov subpsace if reset iteration
            if(np.mod(it,nreset) == 0):
                K = x.copy()
                HK = np.empty((dim,0))
                continue

            # Otherwise, add residuals to Krylov space
            for i in range(n):
                ri = r[:,i]
                if(residuals[i] > tol):
                    prec = e[i] - Qdiag
                    v_new = ri / prec
                    # Perform Gram-Schmidt orthogonalisation twice
                    v_new = v_new - K @ (K.T @ v_new)
                    v_new = v_new - K @ (K.T @ v_new)
                    # Add vector to Krylov subspace if norm is non-vanishing
                    nv = np.linalg.norm(v_new)
                    if(np.linalg.norm(v_new) > 1e-10):
                        v_new = v_new / np.linalg.norm(v_new)
                    K = np.column_stack([K, v_new])
            
        return e, x
