#!/usr/bin/python3
import datetime
import numpy as np
from quantel.utils.linalg import orthogonalise

np.set_printoptions(linewidth=1000,precision=6,suppress=True)
class Davidson:
    """Class to solve lowest eigenvalues and eigenvectors using the Davidson algorithm"""
    def __init__(self, **kwargs):
        """Initialise the Davidson instance"""
        self.control = dict()
        self.control["nreset"] = 50
        self.control["basis_per_root"] = 4
        self.control["collapse_per_root"] = 2

        for key in kwargs:
            if not key in self.control.keys():
                print("ERROR: Keyword [{:s}] not recognised".format(key))
            else: 
                self.control[key] = kwargs[key]

    def run(self, fun_Hv, diag, n, xguess=None, maxit=100, tol=1e-5, plev=1, Hv_args=dict()):
        """ Compute the lowest n eigenvectors and eigenvalues of the Hessian using the 
            Davidson algorithm. 

            An initial guess for the eigenvectors can be provided by the optional argument xguess.
        """
        # Save initial time
        kernel_start_time = datetime.datetime.now()

        if plev>0: print()
        if plev>0: print( "  Initializing Davidson diagonalisation...")
        if plev>0:
            print(f"    > Max iterations = {maxit: 6d}")
            print(f"    > Target states  = {n: 6d}")
            print(f"    > Convergence    = {tol: 6.3e}")
            print(f"    > Reset interval = {self.control['nreset']: 6d}")
            print(f"    > Basis per root = {self.control['basis_per_root']: 6d}")
            print(f"    > Max basis size = {self.control['basis_per_root']*n: 6d}")
        

        # Initialise Krylov subspace
        dim = diag.size
        self.max_subspace  = n * self.control["basis_per_root"]
        self.collapse_size = n * self.control["collapse_per_root"]
        K   = np.empty((dim,0))

        if(n > dim):
            # If the number of requested states exceeds the dimension of the matrix, reset the values
            n = dim
            if(xguess is not None):
                print(f"WARNING: Number of requested eigenvectors exceeds matrix dimension, selecting first {n:6d} guess vectors")
                xguess = xguess[:,:n]
        
        # If no guess provided, start with identity
        if(xguess is None):
            K = np.random.uniform(-1.0,1.0,size=(dim,n))
        else:
            assert(xguess.shape[1] == n)
            K = xguess.copy()
        K = orthogonalise(K,fill=False)

        # Initialise HK vectors
        HK = np.empty((dim, 0))

        # Make K in fortran column-major
        K = np.asfortranarray(K)

        if plev>1: print("  =========================================")
        if plev>1: print("    Step   Max(|res|)    # Conv    Subspace")
        if plev>1: print("  =========================================")

        # Loop over iterations
        comment = ""
        converged = False
        for it in range(maxit+1):
            # Form new HK vectors as required
            for ik in range(HK.shape[1],K.shape[1]):
                # Get step 
                # NOTE this requires column-major ordering for effective slicing
                sk = K[:,ik].copy()
                # Get approximate Hessian on vector
                H_sk = fun_Hv(sk,**Hv_args)
                # Add to HK space
                HK = np.column_stack([HK,H_sk.copy()]) 

            # Solve Krylov subproblem
            A = K.T @ HK
            e, y = np.linalg.eigh(A)
            # Extract relevant eigenvalues (e) and eigenvectors (x)
            e = e[:n]
            x = K @ y[:,:n]
            
            # Compute residuals
            r = HK @ y[:,:n] - x @ np.diag(e)
            residuals = np.max(np.abs(r),axis=0)
            maxres = np.max(residuals)
            nconv  = np.sum(residuals < tol)

            if plev>1:  print(f"  {it: 5d}    {maxres:10.2e}    {nconv: 5d}  {K.shape[1]: 8d}  {comment}")
            # Check convergence
            if all(res < tol for res in residuals):
                converged = True
                break

            # Reset Krylov subpsace if reset iteration
            comment = ""
            #if(np.mod(it+1,self.control["nreset"]) == 0):
            #    K = x.copy()
            #    HK = np.empty((dim,0))
            #    comment = "reset"
            #    continue

            # Limit size of subspace
            if(K.shape[1] + n > self.max_subspace):
                K  = K @ y[:,:self.collapse_size]
                HK = HK @ y[:,:self.collapse_size]

            # Otherwise, add correction vectors to Krylov space
            for i in range(n):
                ri = r[:,i]
                if(residuals[i] > tol):
                    # Build correction vector and normalise
                    v_new = np.zeros(dim)
                    for j in range(dim):
                        denom = e[i] - diag[j]
                        if(abs(denom) > 1e-6):
                            v_new[j] = ri[j] / denom
                    v_new = v_new / np.linalg.norm(v_new)
                    # Perform Gram-Schmidt orthogonalisation twice
                    v_new = v_new - K @ (K.T @ v_new)
                    v_new = v_new - K @ (K.T @ v_new)
                    # Add vector to Krylov subspace if norm is non-vanishing
                    nv = np.linalg.norm(v_new)
                    if(nv > 1e-7):
                        K = np.column_stack([K, v_new / nv])

        if plev>1: print("  =========================================")
        # Save end time and report duration
        kernel_end_time = datetime.datetime.now()
        computation_time = (kernel_end_time - kernel_start_time).total_seconds()
        if(not converged):
            if plev>0: print(f"  Davidson diagonalisation failed to converge in {it: 6d} iterations ({computation_time: 6.2f} seconds)")
        else:
            if plev>0: print(f"  Davidson diagonalisation converged in {it: 6d} iterations ({computation_time: 6.2f} seconds)")
            
        if plev>0: 
            print()
            print("   --------------------------")
            print("    Converged eigenvalues:   ")
            print("   --------------------------")
            for iv, ev in enumerate(e):
                print(f"   {iv: 5d}  { ev: 16.8f}   ")
            print("   --------------------------")
        return e, x
