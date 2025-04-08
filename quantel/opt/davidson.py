#!/usr/bin/python3
import datetime, sys
import numpy as np
from quantel.utils.linalg import orthogonalise  

class Davidson:
    """Class to solve lowest eigenvalues and eigenvectors using the Davidson algorithm"""
    def __init__(self, **kwargs):
        """Initialise the Davidson instance"""
        self.control = dict()
        self.control["nreset"] = 50

        for key in kwargs:
            if not key in self.control.keys():
                print("ERROR: Keyword [{:s}] not recognised".format(key))
            else: 
                self.control[key] = kwargs[key]

    def run(self, fun_Hv, diag, n, xguess=None, maxit=100, tol=1e-5, plev=1, Hv_args=dict()):
        """ Compute the lowest n eigenvectors and eigenvalues of the Hessian using the 
            Davidson algorithm. 

            The Hessian vector product H @ v is computed approximately using the forward finite 
            difference approach given by Eq. (12) in
              Y. L. A. Schmerwitz, G. Levi, H. Jonsson
              J. Chem. Theory Cmput. 19, 3634 (2023)

            An initial guess for the eigenvectors can be provided by the optional argument xguess.
        """
        # Save initial time
        kernel_start_time = datetime.datetime.now()

        if plev>0: print()
        if plev>0: print( "  Initializing Davidson diagonalisation...")
        if plev>0:
            print(f"    > Max iterations = {maxit: 6d}")
            print(f"    > Number states  = {n: 6d}")
            print(f"    > Convergence    = {tol: 6.3e}")

        # Initialise Krylov subspace
        dim = diag.size
        K   = np.empty((dim,0))

        if(n > dim):
            # If the number of requested states exceeds the dimension of the matrix, reset the values
            n = dim
            if(xguess is not None):
                print(f"WARNING: Number of requested eigenvectors exceeds matrix dimension, selecting first {n:6d} guess vectors")
                xguess = xguess[:,:n]
        
        # If no guess provided, start with identity
        if(xguess is None):
            inds = np.argsort(diag)[:n]
            K = 0.4 * (np.random.rand(dim,n)-1)
            for i,j in enumerate(inds):
                K[j,i] = 1
        else:
            assert(xguess.shape[1] == n)
            K = xguess.copy()
        K = orthogonalise(K,np.identity(dim),fill=False)

        # Initialise HK vectors
        HK = np.empty((dim, 0))

        # Make K in fortran column-major
        K = np.asfortranarray(K)

        if plev>1: print("  =========================================")
        if plev>1: print("    Step   Max(|res|)    # Conv            ")
        if plev>1: print("  =========================================")

        # Loop over iterations
        comment = ""
        converged = False
        for it in range(maxit+1):
            # Form new HK vectors as required
            for ik in range(HK.shape[1],K.shape[1]):
                # Get step 
                # NOTE this requires column-major ordering for effective slicing
                sk = K[:,ik]
                # Get approximate Hessian on vector
                H_sk = fun_Hv(sk,**Hv_args)
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
            maxres = np.max(residuals)
            nconv  = np.sum(residuals < tol)

            if plev>1:  print(f"  {it: 5d}    {maxres:10.2e}    {nconv: 5d}  {comment}")
            # Check convergence
            if all(res < tol for res in residuals):
                converged = True
                break

            # Reset Krylov subpsace if reset iteration
            comment = ""
            if(np.mod(it+1,self.control["nreset"]) == 0):
                K = x.copy()
                HK = np.empty((dim,0))
                comment = "reset"
                continue

            # Otherwise, add residuals to Krylov space
            for i in range(n):
                ri = r[:,i]
                if(residuals[i] > tol):
                    v_new = ri / (e[i] - diag + 1e-4)
                    # Perform Gram-Schmidt orthogonalisation twice
                    v_new = v_new - K @ (K.T @ v_new)
                    v_new = v_new - K @ (K.T @ v_new)
                    # Add vector to Krylov subspace if norm is non-vanishing
                    nv = np.linalg.norm(v_new)
                    if(nv > 1e-10):
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
