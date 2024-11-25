class LineSearch:
    """Class to implement a simple Brent line search algorithm"""
    def __init__(self,debug=False):
        self.reset()
        self.debug = debug
    
    def reset(self, eref=0, xref=0, gref=0):
        self.iteration = 0
        self.xl, self.gl, self.el = xref, gref, eref
        self.xr, self.gr, self.er = 1e3, 1e3, 1e3
    
    def next_iteration(self, wolfe1, wolfe2, enext, xnext, gnext):
        """Take the next iteration of the line search"""
        if(self.debug):
            print("Line search debug information")
            print(f"   left     : x = {self.xl: 12.6f}, g = {self.gl: 12.6f}, e = {self.el: 12.6f}")
            print(f"   next     : x = {xnext: 12.6f}, g = {gnext: 12.6f}, e = {enext: 12.6f}")
            print(f"   right    : x = {self.xr: 12.6f}, g = {self.gr: 12.6f}, e = {self.er: 12.6f}")
        
        if(not wolfe1):
            # Overstep, so interpolate
            xnew = (-self.gl*(xnext - self.xl)*(xnext - self.xl)/(2*(enext-self.el) - self.gl*(xnext-self.xl)))
            # Update bracket
            self.xr, self.gr, self.er = xnext, gnext, enext
            xlen = self.xr - self.xl
            # Check new points lies in new bracket
            if(xnew > self.xr):
                xnew = self.xr
            elif(xnew < self.xl + 0.1 * xlen):
                xnew = self.xl + 0.1 * xlen
            elif(xnew > self.xr - 0.1 * xlen):
                xnew = self.xr - 0.1 * xlen

        elif(not wolfe2):
            # Understep, so extrapolate
            xnew = (xnext*self.gl + self.xl*gnext)/(self.gl - gnext)
            xlen = xnext - self.xl
            # Check new points lies in new bracket
            if(xnew < self.xl or xnew > self.xl + 10*xlen):
                xnew = self.xl + 10 * xlen
            elif(xnew < self.xl + 3 * xlen):
                xnew = self.xl + 3 * xlen
            # Update the bracket
            self.xl, self.gl, self.el = xnext, gnext, enext
            # Check inside the right bracke
            if(xnew > self.xr - 0.1 * (self.xr - xnext)):
                xnew = self.xr - 0.1 * (self.xr - xnext)
        
        self.iteration += 1
        return xnew
