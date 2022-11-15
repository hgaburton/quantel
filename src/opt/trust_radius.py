#!/usr/bin/python3

import numpy as np

class TrustRadius:
    '''A class to manage the trust radius updates and steps'''

    def __init__(self, rtrust, minstep, maxstep, dogleg=True):
        '''Initialise the TrustRadius object

           Parameters:
               rtrust : float
                   Initial value of the trust radius
               minstep : float
                   Minimum allowed step size
               maxstep : float
                   Maximum allowed step size
        '''
     
        # Check consistency of input
        assert(minstep <= maxstep)
        assert(rtrust >= minstep and rtrust <= maxstep)

        # Save control variables
        self.__rtrust  = rtrust
        self.__maxstep = maxstep
        self.__minstep = minstep
        self.__eta = 0.125
        self.__dogleg = dogleg

    @property
    def rtrust(self):
        return self.__rtrust

    @property
    def maxstep(self):
        return self.__maxstep

    @property
    def minstep(self):
        return self.__minstep

    @minstep.setter
    def minstep(self, minstep):
        self.__minstep = minstep

    def dogleg_step(self, pu, pb):
        '''Compute an optimal dogleg step for the trust region
        
           Based on the dogleg method described on page 73 in
           "Numerical Optimization", J. Nocedal and S. J. Wright

           Parameters:
               pu : ndarray
                   Unconstrained step, the minimum of the model function along the 
                   steepest-descent pathway
               pb : ndarray
                   Proposed second-order step (either Quasi-Newton or Newton-Raphson)
           
           Returns:
               step : ndarray
                   The optimal dogleg step
               comment : str
                   Descriptor for the chosen type of step
        '''
#        return self.truncated_step(pb)

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

    def truncated_step(self, pu):
        step = np.clip(pu, -self.__rtrust, self.__rtrust)
        lu = np.linalg.norm(pu)
        if(np.linalg.norm(step) == np.linalg.norm(step)):
            return step, "Truncated"
        else:
            return step, ""

        if(lu > self.__rtrust):
            step = (self.__rtrust / lu) * pu
            comment = "Truncated step"
        else:
            step = pu
            comment = ""
        return step, comment


    def accept_step(self, dE_actual, dE_model, step_length):
        '''Determine whether to accept or reject a step and update trust radius
        
           Based on Algorithm 4.1 outline on page 69 of 
           "Numerical Optimization", J. Nocedal and S. J. Wright
           
           Parameters:
               dE_actual : float
                   True energy change for the step
               dE_model : float 
                   Energy change predicted by the trust region quadratic model
               step_length : float
                   Length of the step

           Returns:
               accept : bool
           '''
        # Get quality metric
        rho = dE_actual / dE_model

        #print("{: 10.6f} {: 10.6f} {: 10.6f}".format(dE_model, dE_actual, rho))

        # Update trust radius if suitable
        if rho < 0.25:
            # Reduce trust radius if bad energy model
            self.__rtrust = max(0.5 * self.__rtrust, self.__minstep)
        else:
            # Increase trust radius if good energy model and step length 
            # equals trust radius
            if rho > 0.75 and abs(step_length - self.__rtrust) < 1e-8:
                self.__rtrust = min(2 * self.__rtrust, self.__maxstep)

        # Accept step if trust radius is equal to minimum step size
        if abs(self.__rtrust - self.__minstep) < 1e-8:
            return True
        
        # Otherwise, accept if rho is above a threshold
        return rho > self.__eta
