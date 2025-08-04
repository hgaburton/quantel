#!/usr/bin/python3

import datetime, sys
import numpy as np
from .line_search import LineSearch
from scipy.sparse.linalg import gmres


class Linear:
    def __init__(self, **kwargs):
        self.control = dict()
        self.control["maxstep"] = 0.25  

        for key in kwargs:
            if not key in self.control.keys():
                print("ERROR: Keyword [{:s}] not recognised".format(key))
            else: 
                self.control[key] = kwargs[key]
        
        self.ls = LineSearch()

    def run(self, obj, thresh=1e-6, maxit=100):
        start_time = datetime.datetime.now()
        mat_H = obj.mat_H
        converged = False
        reset = False


        for istep in range(maxit+1):
            e_cur = obj.energy
            grad = obj.gradient
            eta = obj.wfn_grad.copy()
            conv = np.linalg.norm(grad,ord=np.inf)

            # print(f"trial energy {e_cur}")

            if conv < thresh:
                converged = True
                break

            if(istep == 0 or reset):
                # Define state for first step
                wolfe1 = True
                wolfe2 = True
                reset = False
            else:
                wolfe1 = (e_cur - e_ref) <= 1e-4 * np.dot(step,grad_ref) 
                wolfe2 = - np.dot(step,grad) <= - 0.9 * np.dot(step,grad_ref)
                if(self.ls.iteration > 10):
                    wolfe1, wolfe2 = True, True

            
            if (wolfe1 and wolfe2):
                obj.save_last_step()

                print(f"step energy {e_cur}")
                # save new gradient and energy
                grad_ref = grad.copy()
                e_ref = e_cur

                # take step
                step = self.get_linear_step(e_cur, grad, eta, mat_H)
                lscale = self.control["maxstep"] / np.max(np.abs(step))
                if(lscale < 1):
                    step = lscale * step

                # reset line search
                x_ref = 0
                g_ref = np.dot(step,grad) / np.linalg.norm(step)
                self.ls.reset(e_ref,x_ref,g_ref)
            else:
                if(not wolfe2):
                    print("understep")
                elif(not wolfe1):
                    print("overstep")
                # line search parameters
                x_cur = np.linalg.norm(step)
                g_cur = np.dot(step,grad) / x_cur

                # restore last step
                obj.restore_last_step()

                # get next step
                x_next = self.ls.next_iteration(wolfe1, wolfe2, e_cur, x_cur, g_cur)
                step = x_next / x_cur * step

            obj.take_step(step)
        
        end_time = datetime.datetime.now()
        print(f"Time elapsed = {(end_time-start_time).total_seconds()}")
        print(f"Iterations = {istep}")
        return converged
        



    def get_linear_step(self, e_cur, grad, eta, mat_H):
        alpha_diag = 0.2
        H = eta.T @ (mat_H @ eta)
        S = eta.T @ eta
        A = H - e_cur * S
        alpha = - min(0,np.linalg.eigvalsh(A)[0]) + alpha_diag
        M = A + alpha * S
        x = gmres(M, -0.5*grad)[0]
        return x