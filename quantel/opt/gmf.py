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
        self.control["maxstep"] = 0.1
        self.control["rtrust"]  = 0.15
        self.control["hesstol"] = 1e-16

        for key in kwargs:
            if not key in self.control.keys():
                print("ERROR: Keyword [{:s}] not recognised".format(key))
            else: 
                self.control[key] = kwargs[key]

    def run(self, obj, thresh=1e-8, maxit=100, index=0, plev=1, nss=20):
        ''' This function is the one that we will run the Newton-Raphson calculation for a given NR_CASSCF object '''
        kernel_start_time = datetime.datetime.now() # Save initial time

        # Canonicalise, might put Fock matrices in more diagonal form
        obj.canonicalize()

        if plev>0: print()
        if plev>0: print( "  Initializing Eigenvector Following...")
        if plev>0 and (not index == None): print(f"    Target Hessian index = {index: 5d}") 

        # Initialise reference energy
        eref = obj.energy
        dim = obj.dim
        grad = obj.gradient
        gmod, evec = self.get_gmf_gradient(obj,grad,index)

        print(f"  # MOs        = {obj.nmo: 6d}")
        print(f"  # parameters = {obj.dim: 6d}")

        # Initialise lists for subspace vectors
        v_step = []
        v_grad = [gmod]

        if plev>0: print("  ================================================================")
        if plev>0: print("       {:^16s}    {:^8s}    {:^8s}".format("   Energy / Eh","Step Len","Error"))
        if plev>0: print("  ================================================================")

        converged = False
        for istep in range(maxit+1):
            # Get gradient and check convergence
            conv = np.linalg.norm(grad) * np.sqrt(1.0/grad.size)
            eref = obj.energy

            #hess_analytic = obj.hessian
            cur_hind = 0 #np.sum(np.linalg.eigvalsh(hess_analytic)<0)

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
            if(np.dot(qn_step, v_grad[-1]) > 0):
                print("Step has positive component along gradient... reversing direction")
                qn_step *= -1

            # Apply damping 
            alpha = 1 if (istep == 0) else 0.02
#            print(alpha, self.backtrack(obj, qn_step, grad))
            step = alpha * qn_step
            step = np.clip(step,-self.control["maxstep"],self.control["maxstep"])
            comment = ""
  
            # Check for step length converged
            step_length = np.linalg.norm(step)
            if(step_length < thresh*thresh):
                return True

            # Take the step
            obj.take_step(step)
            # Save step
            v_step.append(step.copy())
            # Parallel transport vectors
            v_grad = [obj.transform_vector(v, 0.5 * step) for v in v_grad] 
            v_step = [obj.transform_vector(v, 0.5 * step) for v in v_step] 
            # Compute new GMF gradient (need to parallel transport Hessian eigenvector)
            xguess = np.empty((dim,index))
            for i in range(index):
                xguess[:,i] = obj.transform_vector(evec[:,i], 0.5 * step)
            # Save gradient
            grad = obj.gradient
            gmod, evec = self.get_gmf_gradient(obj,grad,index,xguess=evec)
            v_grad.append(gmod.copy())

            # Remove oldest vectors if subspace is saturated
            if(len(v_step)>nss):
                v_grad.pop(0)
                v_step.pop(0)

            # Increment the iteration counter
            istep += 1

        if plev>0: print("  ================================================================")
        if(converged):
            print("Outcome = {:6d} {: 16.10f} {:6.4e} {:6d}".format(np.sum(np.linalg.eigvalsh(obj.hessian)<0), obj.energy, np.linalg.norm(obj.gradient), istep))
        else:
            print("Outcome = failed")
        kernel_end_time = datetime.datetime.now() # Save end time
        computation_time = kernel_end_time - kernel_start_time
        if plev>0: print("  Eigenvector-following walltime: ", computation_time.total_seconds(), " seconds")

        return converged

    def get_gmf_gradient(self,obj,grad,n,xguess=None):
        """ Compute modified gradient for n-index saddle point search using generalised mode following"""
        if(n==0):
            return grad, None
        # Compute n lowest eigenvalues
        e, x = self.get_Hessian_eigenpairs(obj,grad,n,xguess=xguess)

        # Then project gradient as required
        if(e[n-1] < 0):
            gmod = grad - 2 * x @ (x.T @ grad)
        else:
            for i in range(n):
                if(e[i] >= 0):
                    gmod = grad - x[:,i] @ np.dot(x[:,i], grad)

        return gmod, x
    
    def get_lbfgs_step(self,v_grad,v_step): 
        """Compute the L-BFGS step from gradient and steps"""
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
        r = q
        for i in range(nvec):
            beta = rho[i] * np.dot(yk[i], r)
            r = r + sk[i] * (alpha[i] - beta) 
        return -r

    def backtrack(self, obj, p, grad):
        """Compute optimal step length with backtracking for sufficient decrease"""
        scale = 0.1
        alpha = 1
        c = 1e-4

        obj.save_last_step
        f0 = obj.energy
        sg = c * np.dot(p, grad)
        while 1:
            obj.take_step(alpha * p)
            f = obj.energy
            obj.restore_last_step()
            if(f <= f0 + sg * alpha):
                break
            alpha = alpha * scale
        return alpha

    def get_sr1_hess(self, ssvec, nvec, hess_diag, tol=1e-8):
        """Compute the SR1 Hessian approximation in the subspace"""
        # Access relevant memory      
        ss_grads = ssvec[:,:nvec+1]
        ss_steps = ssvec[:,nvec+1:]
        
        Bk = np.diag(hess_diag)
        for k in range(nvec):
            sk = ss_steps[:,k]
            yk = ss_grads[:,k+1] - ss_grads[:,k]
            rk = yk - Bk @ sk
            rho = np.dot(sk,rk)

            if(abs(rho) >= tol * np.linalg.norm(sk) * np.linalg.norm(rk)):
                Bk = Bk + np.outer(rk, rk) / rho
                
        return Bk

    def get_Hessian_eigenpairs(self,obj,g0,n,xguess=None,max_iter=100,eps=1e-5,tol=1e-2,nreset=50):
        # Initialise Krylov subspace
        dim = obj.dim
        K   = np.empty((dim, 0))
        
        # Get approximate diagonal of Hessian
        Qdiag = obj.get_preconditioner()

        # If no guess provided, start with identity
        if(xguess is None):
            K = np.column_stack([K,np.identity(dim)[:,:n]]) 
            K += 0.01 * np.random.rand(dim,n)
        else:
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
                obj.take_step(eps * sk)
                g1 = obj.gradient.copy()
                obj.restore_last_step()
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
                    # TODO: Get a suitable preconditioner
                    prec = e[i] - Qdiag
                    v_new = ri / prec
                    v_new = v_new - K @ (K.T @ v_new)
                    v_new = v_new - K @ (K.T @ v_new)
                    nv = np.linalg.norm(v_new)
                    if(np.linalg.norm(v_new) > 1e-10):
                        v_new = v_new / np.linalg.norm(v_new)
                    K = np.column_stack([K, v_new])
            
        return e, x
