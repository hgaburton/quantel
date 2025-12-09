import numpy as np
import datetime, sys

# Code to implement the DIIS algorithm for SCF convergence
class DIIS:
    def __init__(self):
        self.err_vecs = []
        self.fock_list = []
    
    def run(self, obj, thresh=1e-6, maxit=100, index=0, plev=1, max_vec=6):
        """Run the DIIS optimisation"""
        self.nbsf = obj.nbsf
        self.max_vec = max_vec

        kernel_start_time = datetime.datetime.now()
        if plev>0: print()
        if plev>0: print("  Running DIIS optimisation...")
        if plev>0:
            print(f"    > Num. MOs          = {obj.nmo: 6d}")
            print(f"    > Max DIIS subspace = {max_vec: 6d}")
        if plev>0: print("  ==========================================")
        if plev>0: print("       {:^16s}    {:^8s}".format("   Energy / Eh","Error"))
        if plev>0: print("  ==========================================")
        converged = False

        for istep in range(maxit+1):
            # Get Fock matrix
            obj.get_fock()
            # Get error vector
            errvec, err = obj.get_diis_error()
            # Print status
            if plev > 0:
                print(" {: 5d} {: 16.10f}    {:8.2e}".format(istep, obj.energy, err))
            sys.stdout.flush()

            if(err < thresh):
                converged = True
                break
            # Append error vector to list
            self.err_vecs.append(errvec)
            # Append Fock matrix to list
            self.fock_list.append(obj.fock)
            # Remove oldest error vector and Fock matrix if we have too many
            if len(self.err_vecs) > self.max_vec:
                self.err_vecs.pop(0)
                self.fock_list.pop(0)

            # Perform DIIS extrapolation
            if len(self.err_vecs) >= 1:
                new_fock = self.diis_extrapolate()
                obj.try_fock(new_fock)
        
        if plev>0: print("  ==========================================")
        kernel_end_time = datetime.datetime.now() # Save end time
        computation_time = (kernel_end_time - kernel_start_time).total_seconds()
        if(not converged):
            if plev>0: print(f"  DIIS failed to converge in {istep: 6d} iterations ({computation_time: 6.2f} seconds)")
        else:
            if plev>0: print(f"  DIIS converged in {istep: 6d} iterations ({computation_time: 6.2f} seconds)")
        sys.stdout.flush()
        return converged


    def diis_extrapolate(self):
        """Perform DIIS extrapolation to get the next Fock matrix"""
        # Get number of error vectors
        nerr = len(self.err_vecs)

        # Get the error matrix
        B = np.zeros((nerr+1, nerr+1))
        for i in range(nerr):
            for j in range(nerr):
                B[i,j] = np.dot(self.err_vecs[i], self.err_vecs[j])
        B[-1,:] = -1
        B[:,-1] = -1
        B[-1,-1] = 0
        
        # Get the right hand side vector
        rhs = np.zeros(nerr+1)
        rhs[-1] = -1
        
        # Solve the linear equations
        coeffs = np.linalg.solve(B, rhs)
        
        # Get the new Fock matrix
        fock = np.zeros_like(self.fock_list[0])
        for i in range(nerr):
            fock += coeffs[i] * self.fock_list[i]
        return fock