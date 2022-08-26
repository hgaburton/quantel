#!/usr/bin/python3

import numpy as np

class TrustRadius:
    '''A class to manage the trust radius updates and steps'''

    def __init__(self, rtrust, minstep, maxstep):
     
        # Check consistency of input
        assert(minstep < maxstep)
        assert(rtrust > minstep and rtrust < maxstep)

        # Save control variables
        self.__rtrust  = rtrust
        self.__maxstep = maxstep
        self.__minstep = minstep
        self.__eta = 0.125

    @property
    def rtrust(self):
        return self.__rtrust

    @property
    def maxstep(self):
        return self.__maxstep

    @property
    def minstep(self):
        return self.__minstep

    def dogleg_step(self, pu, pb):
        '''Compute a dogleg step using the algorithm in...'''

        # Get lengths of Newton-Rahpson (pb) and Steepest-Descent (pu) steps 
        lb = np.linalg.norm(pb)
        lu = np.linalg.norm(pu)

        if(lb <= self.__rtrust):
            # Take Newton step if within trust radius
            step    = pb
            comment = ""
        elif (lu >= self.__rtrust):
            # Take scaled steepest-descent step if pu longer than trust radius
            step    = (self.__rtrust / lu) * pu
            comment = "Gradient step"
        else:
            # Get dogleg vector
            dl = pb - pu

            # Get step length by solving quadratic
            a = np.dot(dl, dl)
            c = lu * lu - self.__rtrust * self.__rtrust
            b = 2.0 * np.dot(pu, dl)
            
            # Get the step length and dogleg step
            tau  = (- b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
            step = pu + tau * (pb - pu)
            comment = "Dogleg step"

        return step, comment


    def accept_step(self, dE_actual, dE_model, step_length):
        '''Determine whether to accept or reject a step and update the 
           trust radius accordingly'''
        
        # Get quality metric
        rho = dE_actual / dE_model

        # Update trust radius if suitable
        if rho < 0.25:
            # Reduce trust radius if bad energy model
            self.__rtrust = max(0.25 * self.__rtrust, self.__minstep)
        else:
            # Increase trust radius if good energy model and step length 
            # equals trust radius
            if rho > 0.75 and abs(step_length - self.__rtrust) < 1e-8:
                self.__rtrust = min(2.0 * self.__rtrust, self.__maxstep)

        # Accept step if trust radius is equal to minimum step size
        if abs(self.__rtrust - self.__minstep) < 1e-8:
            return True
        
        # Otherwise, accept if rho is above a threshold
        return rho > self.__eta
