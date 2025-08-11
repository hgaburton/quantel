#!/usr/bin/python3

import datetime, sys
import numpy as np
from .line_search import LineSearch
from .trust_radius import TrustRadius
from scipy.sparse.linalg import gmres


class Linear:
    '''
    Class to implement the linear method.
    Based on J. Phys. Chem. A 2024, 128, 40, 8762–8776

    There are two numerical optimisation methods:
        - run_linesearch() : uses a line search algorithm 
        - run_dogleg() : uses a Powell's dog leg trust region method
    '''
    def __init__(self, **kwargs):
        self.control = dict()
        self.control["startrad"] = 0.25
        self.control["maxrad"] = 1
        self.control["minrad"] = 1e-5


        for key in kwargs:
            if not key in self.control.keys():
                print("ERROR: Keyword [{:s}] not recognised".format(key))
            else: 
                self.control[key] = kwargs[key]
        
        self.ls = LineSearch(debug=False)
        self.tr = TrustRadius(self.control["startrad"], self.control["minrad"], self.control["maxrad"])


    def run_linesearch(self, obj, thresh=1e-6, maxit=100, strong=False):
        '''
        Minimization using line search.

        thresh : float, threshold for gradient convergence
        maxit : integer, maximum number of iterations
        strong : bool, Enables strong Wolfe conditions if True, weak Wolfe conditions otherwise

        Run the optimisation for a particular objective function obj.
            
            obj must have the following methods implemented:
              + energy
              + gradient
              + obj.wfn_grad : tangent vectors of the wavefunction
              + dim
              + get_preconditioner()
              + take_step()
        '''
        start_time = datetime.datetime.now()
        mat_H = obj.mat_H
        converged = False
        reset = False

        for istep in range(maxit+1):
            # get energy, gradient, tangent vectors(eta), preconditioner and check convergence
            e_cur = obj.energy
            grad = obj.gradient
            eta = obj.wfn_grad.copy()
            prec = obj.get_preconditioner()
            conv = np.linalg.norm(grad,ord=np.inf)

            if istep > 0:
                print(" {: 5d} {: 16.10f}    {:8.2e}     {:8.2e}    {:10s}".format(
                      istep, e_cur, np.max(np.abs(step)), conv, comment))
            else:
                print(" {: 5d} {: 16.10f}                 {:8.2e}".format(istep, e_cur, conv))
            sys.stdout.flush()

            if conv < thresh:
                converged = True
                break

            if(istep == 0 or reset):
                # Define state for first step
                wolfe1 = True
                wolfe2 = True
                reset = False
            else:
                # check Wolfe conditions
                phi = np.dot(step,grad_ref) 
                wolfe1 = (e_cur - e_ref) <= 1e-4 * phi 
                if strong:
                    wolfe2 = abs(np.dot(step,grad)) <= - 0.9 * phi
                else:
                    wolfe2 = - np.dot(step,grad) <= - 0.9 * phi
                if(self.ls.iteration > 10):
                    # accept if 10 iterations have passed
                    wolfe1, wolfe2 = True, True

            comment = ""
            if (wolfe1 and wolfe2):
                # save step
                obj.save_last_step()

                # get initial a_0 guess from quadratic interpolation
                if istep > 0:
                    a_0 = min(1,2*(e_cur-e_ref)/np.dot(p,grad_ref))
                else:
                    a_0 = 1

                # save new gradient and energy
                grad_ref = grad.copy()
                e_ref = e_cur

                # take step
                p = self.get_linear_step(e_cur, grad, eta, mat_H, prec, adiag=conv*conv)

                # scale p with a_0
                step = a_0 * p

                # Need to make sure s.g < 0 to maintain positive-definite L-BFGS Hessian 
                if(np.dot(step,grad_ref) > 0):
                    # print("  Step has positive overlap with gradient - reversing direction")
                    step *= -1
                    p *= -1
                    reset = True
                    comment = "reversing"
                
                # reset line search
                x_ref = 0
                g_ref = np.dot(step,grad) / np.linalg.norm(step)
                self.ls.reset(e_ref,x_ref,g_ref)
            else:
                if(not wolfe1):
                    comment = "overstep"
                elif(not wolfe2):
                    comment = "understep"
                # line search parameters
                x_cur = np.linalg.norm(step)
                g_cur = np.dot(step,grad) / x_cur

                # restore last step
                obj.restore_last_step()

                # get next step
                x_next = self.ls.next_iteration(wolfe1, wolfe2, e_cur, x_cur, g_cur)
                a_cur = x_next / x_cur
                step = a_cur * step

            obj.take_step(step)
        
        end_time = datetime.datetime.now()
        print(f"Time elapsed = {(end_time-start_time).total_seconds()}s")
        print(f"Iterations = {istep}")
        return converged


    def run_dogleg(self, obj, thresh=1e-6, maxit=100):
        '''
        Minimization using trust region dogleg method

        thresh : float, threshold for gradient convergence
        maxit : integer, maximum number of iterations

        Run the optimisation for a particular objective function obj.
            
            obj must have the following methods implemented:
              + energy
              + gradient
              + obj.wfn_grad : tangent vectors of the wavefunction
              + dim
              + get_preconditioner()
              + take_step()
        '''
        start_time = datetime.datetime.now()
        mat_H = obj.mat_H
        converged = False

        for istep in range(maxit+1):
            # get energy, gradient, tangent vectors(eta), preconditioner and check convergence
            e_cur = obj.energy
            grad = obj.gradient
            eta = obj.wfn_grad.copy()
            prec = obj.get_preconditioner()
            conv = np.linalg.norm(grad,ord=np.inf)

            if istep > 0:
                print(" {: 5d} {: 16.10f}    {:8.2e}     {:8.2e}    {:10s}".format(
                      istep, e_cur, np.max(np.abs(step)), conv, comment))
            else:
                print(" {: 5d} {: 16.10f}                 {:8.2e}".format(istep, e_cur, conv))
            sys.stdout.flush()

            if conv < thresh:
                converged = True
                break

            if(istep == 0):
                # Define state for first step
                accept = True
            else:
                # check step is acceptable
                ared = e_cur - e_ref
                accept = self.tr.accept_step(ared, pred, np.linalg.norm(step))

            comment = ""
            if (accept):
                # save step
                obj.save_last_step()

                # save new gradient and energy
                grad_ref = grad.copy()
                e_ref = e_cur

                # take step
                pb = self.get_linear_step(e_cur, grad, eta, mat_H, prec, adiag=conv*conv)
                pu = - ((np.dot(grad,grad))/(grad.dot(self.B.dot(grad)))) * grad
                step, comment = self.tr.dogleg_step(pu, pb)


            else:
                # restore last step and get energy, gradient, etc again.
                obj.restore_last_step()
                e_cur = obj.energy
                grad = obj.gradient
                eta = obj.wfn_grad.copy()
                prec = obj.get_preconditioner()
                conv = np.linalg.norm(grad,ord=np.inf)

                # get step
                pb = self.get_linear_step(e_cur, grad, eta, mat_H, prec, adiag=conv*conv)
                pu = - ((np.dot(grad,grad))/(grad.dot(self.B.dot(grad)))) * grad
                step, comment = self.tr.dogleg_step(pu, pb)
                comment = "Restore step"

            if(np.dot(step,grad_ref) > 0):
                # print("  Step has positive overlap with gradient - reversing direction")
                step *= -1
                comment = "reversing"

            # get predicted energy change to 2nd order
            pred = (np.dot(grad,step) + step.dot(self.B.dot(step)))

            obj.take_step(step)
        
        end_time = datetime.datetime.now()
        time = (end_time-start_time).total_seconds()
        print(f"Time elapsed = {time: 10.3f}s")
        print(f"Iterations = {istep}")
        return converged
        

    def get_linear_step(self, e_cur, grad, eta, mat_H, prec, adiag=0.1):
        """ Compute the linear step, x, from solving the eignvalue equation:
            Mx = -0.5g

            M = H - e_cur * S + alpha * S
            where alpha is the lowest eigenvalue of H - e_cur * S or 0 plus an adiag term
            
            Based on J. Phys. Chem. A 2024, 128, 40, 8762–8776
        """
        # compute H and S
        Pinv = np.diag(np.power(np.sqrt(np.clip(prec,0.1,None)),-1))
        self.H = eta.T @ (mat_H @ eta)
        S = eta.T @ eta
        # Transform the prec basis (hope alpha is regularized)
        Ht = Pinv @ self.H @ Pinv
        St = Pinv @ S @ Pinv
        # construct matrix A
        A = self.H - e_cur * S
        # get alpha from the lowest eigenvalue if it is positive definite
        adiag = 0.2
        Emin = np.linalg.eigvalsh(A)[0]
        alpha = - min(0,Emin) + adiag

        M = A + alpha * S
        self.B = A
        # get eigenvectors 
        gt = Pinv @ grad
        x = gmres(M, -0.5*grad)[0]
        #x = Pinv @ xt
        return x