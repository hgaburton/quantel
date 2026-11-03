#!/usr/bin/python3

import datetime, sys
import numpy as np
from .trust_radius import TrustRadius
from quantel.utils.linalg import orthogonalise  
import scipy

class SR1:
    """Class to implement limited-memory Symmetric-Rank-1 (l-SR1) optimisation for targeting 
       saddle points. 

       This implementation follows Algorithm 1 in
          G. Levi, A. V. Ivanov, and H. Jonsson, J. Chem. Theory Comput. 16, 6968 (2020)
       with a trust radius approach to truncate the step length.
    """

    def __init__(self, **kwargs):
        """Initialise the l-SR1 instance"""
        self.control = dict()
        self.control["minstep"] = 0.01
        self.control["maxstep"] = np.pi
        self.control["rtrust"]  = 0.15
        self.control["precmin"] = 1e0
        self.control["max_subspace"] = 20

        for key in kwargs:
            if not key in self.control.keys():
                print("ERROR: Keyword [{:s}] not recognised".format(key))
            else: 
                self.control[key] = kwargs[key]

        # Initialise the trust radius controller
        self.__trust = TrustRadius(self.control["rtrust"], self.control["minstep"], 
                                   self.control["maxstep"])

    def run(self, obj, thresh=1e-8, maxit=100, index=0, plev=1):
        """ Run the optimisation for a particular objective function obj.
            
            obj must have the following methods implemented:
              + energy
              + gradient
              + dim
              + take_step()
              + save_last_step()
              + restore_last_step()
              + transform_vector()
              + get_preconditioner(abs=False)
              + canonicalize()
        """
        kernel_start_time = datetime.datetime.now() # Save initial time

        # Canonicalise, might put Fock matrices in more diagonal form
        obj.canonicalize()

        if plev>0: print()
        if plev>0: print( "  Initializing L-SR1 optimisation...")

        # Initialise reference energy
        eref = obj.energy
        dim = obj.dim
        grad = obj.gradient

        # Get parameters
        precmin = self.control["precmin"]
        max_subspace = self.control["max_subspace"]

        if plev>0:
            print(f"    > Num. MOs     = {obj.nmo: 6d}")
            print(f"    > Num. params  = {obj.dim: 6d}")
            print(f"    > Max subspace = {max_subspace: 6d}")
            print(f"    > precmin      = {precmin: 6.2e}")
            print()

        # Initialise lists for the SR1 update vectors j^(k) and scalars rho^(k).
        v_j   = []
        v_rho = []

        if plev>0: print("  ================================================================")
        if plev>0: print("       {:^16s}    {:^8s}    {:^8s}".format(
                         "   Energy / Eh","Step Len","Error"))
        if plev>0: print("  ================================================================")

        converged = False
        for istep in range(maxit+1):
            # Get gradient and check convergence
            grad = obj.gradient
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

            # Get preconditioner, retaining its sign so that B_0 can describe negative curvature.
            # Small elements keep their sign, while undefined (zero) elements are taken as positive
            prec = obj.get_preconditioner(abs=False)
            for i in range(dim):
                if(abs(prec[i]) < precmin):
                    prec[i] = - precmin if prec[i] < 0 else precmin

            # Get L-SR1 quasi-Newton step
            qn_step = - self.apply_inverse_hessian(v_j,v_rho,prec,grad)

            # Apply step truncation
            step, comment = self.__trust.truncated_step_norm(qn_step)

            # Compute model energy change
            alpha = np.dot(step, qn_step) / np.dot(qn_step, qn_step)
            dE_model = (1 - 0.5 * alpha) * np.dot(step, grad)
  
            # Check for step length converged
            step_length = np.linalg.norm(step)
            if(step_length < thresh*thresh):
                return True

            # Take the step
            obj.take_step(step)

            # Save step
            s = step.copy()
            # Get new gradient
            gnew = obj.gradient

            # Get actual energy change
            dE = obj.energy - eref

            # Assess trust radius
            if istep > 0:
                # Save reference energy if we accept step, otherwise undo the step
                if self.__trust.accept_step(dE, dE_model, step_length):
                    eref = obj.energy
                    # Save the new origin 
                    obj.save_last_step()
                    # Parallel transport vectors
                    grad = obj.transform_vector(grad, 0.5 * step)
                    s    = obj.transform_vector(s, 0.5 * step)
                    v_j  = [obj.transform_vector(v, 0.5 * step) for v in v_j]
                else:
                    # Otherwise undo the step
                    comment = "No step"
                    # Parallel transport gradient back to origin
                    gnew = obj.transform_vector(gnew, -0.5 * step)
                    # Restore step
                    obj.restore_last_step()

            # Remove oldest update if subspace is saturated
            if(len(v_j) > 0 and len(v_j) >= max_subspace):
                v_j.pop(0)
                v_rho.pop(0)

            # Compute and store the SR1 update for the new step and gradient change
            y = gnew - grad 
            jk = s - self.apply_inverse_hessian(v_j,v_rho,prec,y)
            jy = np.dot(jk, y)
            rhok = eps if abs(jy) < 1e-12 else 1.0 / jy
            v_j.append(jk)
            v_rho.append(rhok)

            # Increment the iteration counter
            istep += 1

        if plev>0: print("  ================================================================")

        # Save end time and report 
        kernel_end_time = datetime.datetime.now()
        computation_time = kernel_end_time - kernel_start_time
        if plev>0: print("  Limited-memory SR1 walltime: ", 
                         computation_time.total_seconds(), " seconds")

        return converged

    def apply_inverse_hessian(self,v_j,v_rho,prec,vec):
        """ Apply the SR1 inverse Hessian to a vector using eq 19 in Levi et al.

            The inverse Hessian is represented as
                B^(k) = B_0^(k) + sum_i rho^(i) j^(i) j^(i)T
            where B_0^(k) is the inverse of the latest diagonal preconditioner, and j^(i) and
            rho^(i) are stored from previous iterations so only vector-vector products are required.
        """
        out = vec / prec
        for i in range(len(v_j)):
            out = out + v_rho[i] * np.dot(v_j[i], vec) * v_j[i]
        return out
