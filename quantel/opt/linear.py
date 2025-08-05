#!/usr/bin/python3

import datetime, sys
import numpy as np
from .line_search import LineSearch
from .trust_radius import TrustRadius
from scipy.sparse.linalg import gmres


class Linear:
    '''
    Based on J. Phys. Chem. A 2024, 128, 40, 8762–8776
    '''
    def __init__(self, **kwargs):
        self.control = dict()
        self.control["maxstep"] = 0.2

        for key in kwargs:
            if not key in self.control.keys():
                print("ERROR: Keyword [{:s}] not recognised".format(key))
            else: 
                self.control[key] = kwargs[key]
        
        self.ls = LineSearch(debug=False)
        self.tr = TrustRadius(0.5, 0.1, 1)


    def run(self, obj, thresh=1e-6, maxit=100):
        '''
        Minimization using line search
        '''
        start_time = datetime.datetime.now()
        mat_H = obj.mat_H
        converged = False
        reset = False
        a_cur = 1


        for istep in range(maxit+1):
            e_cur = obj.energy
            grad = obj.gradient
            eta = obj.wfn_grad.copy()
            prec = obj.get_preconditioner()
            conv = np.linalg.norm(grad,ord=np.inf)

            if istep > 0:
                print(" {: 5d} {: 16.10f}    {:8.2e}     {:8.2e}    {:10s}".format(
                      istep, e_cur, np.max(np.abs(step)), conv, comment))
                #print(f" {istep: 5d} {ecur: 16.10f}   {ecur-eref: 8.2e}   {RMSDP: 8.2e}   {MaxDP: 8.2e}   {MaxGrad: 8.2e}")
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
                goldstein = True
                reset = False
            else:
                wolfe1 = (e_cur - e_ref) <= 1e-4 * np.dot(step,grad_ref) 
                wolfe2 = - np.dot(step,grad) <= - 0.9 * np.dot(step,grad_ref)
                goldstein = (e_cur - e_ref) >= (1-0.1) * np.dot(step,grad_ref)
                if(np.max(np.abs(step))>=self.control["maxstep"]):
                    # We're on maximum step size, so we can't extrapolate
                    #print("Overriding Wolfe2")
                    wolfe2 = True
                if(self.ls.iteration > 10):
                    wolfe1, wolfe2 = True, True
                    goldstein = True

            comment = ""
            if (wolfe1 and wolfe2):
                # save step
                obj.save_last_step()

                # save new gradient and energy
                grad_ref = grad.copy()
                e_ref = e_cur

                # take step
                step = self.get_linear_step(e_cur, grad, eta, mat_H, prec, adiag=conv*conv)

                # Need to make sure s.g < 0 to maintain positive-definite L-BFGS Hessian 
                if(np.dot(step,grad_ref) > 0):
                    print("  Step has positive overlap with gradient - reversing direction")
                    step *= -1
                    reset = True

                lscale = self.control["maxstep"] / np.max(np.abs(step))
                if(lscale < 1):
                    step = lscale * step
                    comment = "rescaled"

                # reset line search
                x_ref = 0
                g_ref = np.dot(step,grad) / np.linalg.norm(step)
                self.ls.reset(e_ref,x_ref,g_ref)
            else:
                if(not wolfe2):
                    comment = "understep"
                elif(not wolfe1):
                    comment = "overstep"
                # line search parameters
                x_cur = np.linalg.norm(step)
                g_cur = np.dot(step,grad) / x_cur

                # restore last step
                obj.restore_last_step()

                # get next step
                x_next = self.ls.next_iteration(wolfe1, wolfe2, e_cur, x_cur, g_cur)
                #print(f"x_next = {x_next: 10.6f}  x_cur = {x_cur: 10.6f}")
                a_cur = x_next / x_cur
                step = a_cur * step

            obj.take_step(step)
        
        end_time = datetime.datetime.now()
        print(f"Time elapsed = {(end_time-start_time).total_seconds()}")
        print(f"Iterations = {istep}")
        return converged

    def run2(self, obj, thresh=1e-6, maxit=100):
        '''
        Minimization using trust radius dogleg method
        '''
        start_time = datetime.datetime.now()
        mat_H = obj.mat_H
        converged = False
        reset = False

        for istep in range(maxit+1):
            e_cur = obj.energy
            grad = obj.gradient
            eta = obj.wfn_grad.copy()
            prec = obj.get_preconditioner()
            conv = np.linalg.norm(grad,ord=np.inf)

            if istep > 0:
                print(" {: 5d} {: 16.10f}    {:8.2e}     {:8.2e}    {:10s}".format(
                      istep, e_cur, np.max(np.abs(step)), conv, comment))
                #print(f" {istep: 5d} {ecur: 16.10f}   {ecur-eref: 8.2e}   {RMSDP: 8.2e}   {MaxDP: 8.2e}   {MaxGrad: 8.2e}")
            else:
                print(" {: 5d} {: 16.10f}                 {:8.2e}".format(istep, e_cur, conv))
            sys.stdout.flush()

            if conv < thresh:
                converged = True
                break

            if(istep == 0 or reset):
                # Define state for first step
                accept = True
            else:
                ared = e_ref - e_cur
                pred = - (np.dot(grad,step) + 0.5 * step @ (self.B @ step))
                # print(ared, pred)
                accept = self.tr.accept_step(ared, pred, np.linalg.norm(step))

            comment = ""
            if (accept):
                # save step
                obj.save_last_step()

                # save new gradient and energy
                grad_ref = grad.copy()
                e_ref = e_cur

                # take step
                step = self.get_linear_step(e_cur, grad, eta, mat_H, prec, adiag=conv*conv)
                step, comment = self.tr.dogleg_step(-grad*0.01, step)

                lscale = self.control["maxstep"] / np.max(np.abs(step))
                if(lscale < 1):
                    step = lscale * step
                    comment = "rescaled"

                # Need to make sure s.g < 0 to maintain positive-definite L-BFGS Hessian 
                if(np.dot(step,grad_ref) > 0):
                    print("  Step has positive overlap with gradient - reversing direction")
                    step *= -1
                    reset = True

            else:
                if not accept:
                    comment = "Restore step"
                # restore last step
                obj.restore_last_step()

                step, comment = self.tr.dogleg_step(-grad, step)
                
                if(lscale < 1):
                    step = lscale * step
                    comment = "rescaled"
                step, comment = self.tr.dogleg_step(-grad, step)

            obj.take_step(step)
        
        end_time = datetime.datetime.now()
        print(f"Time elapsed = {(end_time-start_time).total_seconds()}")
        print(f"Iterations = {istep}")
        return converged
        

    def get_linear_step(self, e_cur, grad, eta, mat_H, prec, adiag=0.1):
        # compute H and S
        Pinv = np.diag(np.power(np.sqrt(np.clip(prec,0.1,None)),-1))
        H = eta.T @ (mat_H @ eta)
        S = eta.T @ eta
        # Transform the prec basis (hope alpha is regularized)
        Ht = Pinv @ H @ Pinv
        St = Pinv @ S @ Pinv
        # construct matrix A
        A = H - e_cur * S
        # get alpha from the lowest eigenvalue if it is positive definite
        # alpha_diag = 0.2
        Emin = np.linalg.eigvalsh(A)[0]
        alpha = - min(0,Emin) + adiag
        #print(f"alpha = {alpha: 10.8f}   Emin = {Emin: 10.8f}   Emin2 = {Emin2: 10.8f}")
        
        M = A + alpha * S
        self.B = 2 * M
        # get eigenvectors 
        gt = Pinv @ grad
        x = gmres(M, -0.5*grad)[0]
        #x = Pinv @ xt
        return x