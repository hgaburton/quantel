
from logging import config
import numpy
from quantel import wfn
from quantel.opt.davidson import Davidson
from quantel.opt.hybrid_ef import HybridEF
from quantel.opt.lbfgs import LBFGS
import os
import glob


class Network:
    """
    Class to build an energy landscape network. Searches for local minima and connecting transition states
    using single-ended optimisation methods based on hybrid eigenvector-following.
    """    
    def __init__(self, ints, wfntype):
        """
        Docstring for __init__
        
        :param self: Description
        :param ints: Description
        :param config: Description
        """
        self.ints = ints

        # Make directory to store minima and transition states
        os.makedirs("min", exist_ok=True)
        os.makedirs("ts", exist_ok=True)
        # Save wavefunction type
        self.WFN = wfntype

        self.nmin = 0
        self.nts  = 0
        self.mindata = dict()
        self.tsdata  = dict()
        self.connections = set()

        # Read all minima in the database
        self.read_minima()
        # Read all transition states in the database
        self.read_ts()
        # Read all the connections in the database
        self.read_connections()

    def compare_solutions(self, wfn1, wfn2, ethresh=1e-6, sthresh=1e-6):
        """ 
        Check whether two solutions are equivalent.
        """
        if abs(wfn1.energy - wfn2.energy) < ethresh:
            if abs(1.0 - abs(wfn1.overlap(wfn2))) < sthresh:
                return True
        return False

    def check_minimum(self, wfn):
        """
        Check whether a solution is a minimum by reconverging LBFGS and checking the Hessian index.
        
        :param wfn: Wavefunction solution to check
        """
        wfn0 = wfn.copy()
        # Setup the optimiser
        opt = LBFGS()
        # Reconverge the solution
        opt.run(wfn, maxit=1000, thresh=1e-8, plev=0)
        # Check the Hessian index
        wfn.get_davidson_hessian_index(plev=0,approx_hess=False)
        # Check solutions are equiv
        same_solution = self.compare_solutions(wfn0, wfn)
        # Return True if the solution is a minimum, False otherwise
        return wfn.hess_index[0] == 0 and same_solution
    
    def check_ts(self, wfn):
        """
        Check whether a solution is a transition state by reconverging LBFGS and checking the Hessian index.
        
        :param wfn: Wavefunction solution to check
        """
        wfn0 = wfn.copy()
        # Setup the optimiser
        opt = HybridEF()
        # Reconverge the solution
        opt.run(wfn, index=1, maxit=1000, thresh=1e-6, plev=0)
        # Check the Hessian index
        wfn.get_davidson_hessian_index(plev=0,approx_hess=False)
        # Return True if the solution is a transition state, False otherwise
        return wfn.hess_index[0] == 1 and self.compare_solutions(wfn0, wfn)
    
    def is_new_minimum(self, wfn):
        """
        Check whether a minimum solution is new by comparing against previously found minima.
        
        :param wfn: Minimum wavefunction solution to check
        """
        prev=0
        for prev, tag in self.mindata.items():
            otherwfn = self.WFN(self.ints, 'cs')
            otherwfn.read_from_disk(tag)
            if self.compare_solutions(wfn, otherwfn):
                return False, prev, tag
        return True, prev+1, "min/{:04d}".format(prev+1)
    
    def is_new_ts(self, wfn):
        """
        Check whether a transition state solution is new by comparing against previously found transition states.
        
        :param wfn: Transition state wavefunction solution to check
        """
        prev=0
        for prev, tag in self.tsdata.items():
            otherwfn = self.WFN(self.ints, 'cs')
            otherwfn.read_from_disk(tag)
            if self.compare_solutions(wfn, otherwfn):
                return False, prev, tag
        return True, prev+1, "ts/{:04d}".format(prev+1)

    def is_new_connection(self, its, min1, min2):
        """
        Check whether a connection is new by comparing against previously found connections.
        
        :param its: Transition state index
        :param min1: First minimum index
        :param min2: Second minimum index
        """
        return not ((its, min(min1,min2), max(min1,min2)) in self.connections)

    def save_minimum(self, wfn):
        """
        Save a minimum solution to disk and update the minima data.
        
        :param wfn: Minimum wavefunction solution to save
        """
        self.nmin += 1
        tag = "min/{:04d}".format(self.nmin)
        wfn.save_to_disk(tag)
        self.mindata[self.nmin] = tag
        with open('min.data', 'a') as f:
            f.write(f"{wfn.energy: 20.15f} {1: 20.15f} {1:8d} {1:16.10f} {1:16.10f} {1:16.10f}\n")

    def save_ts(self, wfn, min1, min2):
        """
        Save a transition state solution to disk and update the transition state data.
        
        :param wfn: Transition state wavefunction solution to save
        :param min1: First minimum index
        :param min2: Second minimum index
        """
        self.nts += 1
        tag = "ts/{:04d}".format(self.nts)
        wfn.save_to_disk(tag)
        self.tsdata[self.nts] = tag
        self.save_connection(wfn.energy, self.nts, min1, min2)
    
    def save_connection(self, Ets, its, min1, min2):
        """
        Save a connection to disk and update the connections data.
        
        :param Ets: Transition state energy
        :param its: Transition state index
        :param min1: First minimum index
        :param min2: Second minimum index
        """
        self.connections.add((its, min(min1,min2), max(min1,min2)))
        with open('ts.data', 'a') as f:
            f.write(f"{Ets: 20.15f} {1: 20.15f} {1:8d} {min(min1,min2):8d} {max(min1,min2):8d} {1:16.10f} {1:16.10f} {1:16.10f}\n")

    def read_connections(self):
        """
        Read in connections from disk and update the connections data.
        """
        if(not os.path.isfile('ts.data')):
            return
        for its, line in enumerate(open('ts.data', 'r')):
            data = line.split()
            min1 = min(int(data[3]), int(data[4]))
            min2 = max(int(data[3]), int(data[4]))
            self.connections.add((its+1, min1, min2))

    def read_minima(self):
        """
        Read in minima solutions from disk and update the minima data.
        """
        for old_tag in sorted(glob.glob("min/*.solution")):
            old_tag = old_tag[:-9]
            
            state = self.WFN(self.ints, 'cs')
            state.read_from_disk(old_tag)
            self.nmin += 1
            self.mindata[self.nmin] = old_tag
        print("Found {:d} minima.".format(self.nmin))

    def read_ts(self):
        """
        Read in transition state solutions from disk and update the transition state data.
        """
        for old_tag in sorted(glob.glob("ts/*.solution")):
            old_tag = old_tag[:-9]
            state = self.WFN(self.ints, 'cs')
            state.read_from_disk(old_tag)
            self.nts += 1
            self.tsdata[self.nts] = old_tag
        print("Found {:d} transition states.".format(self.nts))

    def read_solutions(self, prefix="./"):
        """
        Read in solutions and classify them as minima or transition states.
        """
        for old_tag in sorted(glob.glob(prefix+"*.solution")):
            old_tag = old_tag[:-9]
            print("\n  Original solution: ", old_tag)
            state = self.WFN(self.ints, 'cs')
            state.read_from_disk(old_tag)
            state_min = state.copy()
            state_ts  = state.copy()
            if self.check_minimum(state_min):
                print("  Solution is a minimum. E = ", state_min.energy)
                newmin, prev, tag = self.is_new_minimum(state_min)
                if(not newmin):
                    print("  Solution is equivalent to minimum {:d} with tag {:s}".format(prev, tag))
                else:
                    self.save_minimum(state_min)
            elif self.check_ts(state_ts):
                print("  Solution is a transition state. E = ", state_ts.energy)
                newts, prev, tag = self.is_new_ts(state_ts)
                if(not newts):
                    print("  Solution is equivalent to transition state {:d} with tag {:s}".format(prev, tag))
                else:
                    (Emin1, Imin1), (Emin2, Imin2) = self.connect(state_ts)
                    self.save_ts(state_ts, Imin1, Imin2)
            else:
                print("  Solution is neither a minimum nor a transition state.")

    
    def connect(self, wfnts):
        """
        Find the minima connected to a transition state
        """
        e,v = Davidson().run(wfnts.hess_on_vec, wfnts.get_preconditioner(), 2, maxit=1000, plev=0)

        # Take forward step 
        wfn1 = wfnts.copy()
        wfn1.take_step(0.001*numpy.copy(v[:,0]))
        lbfgs = LBFGS()
        converged1 = lbfgs.run(wfn1, maxit=1000, thresh=1e-6,plev=0)
        Emin1 = wfn1.energy
        checkmin1 = converged1 and self.check_minimum(wfn1)
        if checkmin1:
            newmin, min1_i, min1_tag = self.is_new_minimum(wfn1)
            if(not newmin):
                print(f"  Forward minimum is equivalent to minimum {min1_i:<4d} with tag {min1_tag:s} E = {Emin1: 16.10f}")
            else:
                print(f"  Forward minimum is new {self.nmin+1:<4d} with tag min/{self.nmin+1:04d} E = {Emin1: 16.10f}")
                self.save_minimum(wfn1)

        # Take backward step
        wfn2 = wfnts.copy()
        wfn2.take_step(-0.001*numpy.copy(v[:,0]))
        converged2 = lbfgs.run(wfn2, maxit=1000, thresh=1e-6,plev=0)
        Emin2 = wfn2.energy
        checkmin2 = converged2 and self.check_minimum(wfn2)
        if checkmin2:
            newmin, min2_i, min2_tag = self.is_new_minimum(wfn2)
            if(not newmin):
                print(f"  Backward minimum is equivalent to minimum {min2_i:<4d} with tag {min2_tag:s} E = {Emin2: 16.10f}")
            else:
                print(f"  Backward minimum is new {self.nmin+1:<4d} with tag min/{self.nmin+1:04d} E = {Emin2: 16.10f}")
                self.save_minimum(wfn2)
        
        if (not checkmin1) or (not checkmin2):
            return None

        return sorted([(min1_i, Emin1), (min2_i, Emin2)])

    def find_connections(self,stepsize=0.1,cycles=10):
        """
        Find connections between minima via transition states using single-ended transition 
        state searches.
        """
        for icycle in range(cycles):
            print("Cycle {:d}/{:d}".format(icycle+1, cycles))
            current_minima = dict(self.mindata)
            for _, mtag in current_minima.items():
                print(" > Starting  from minimum {}".format(mtag))
                
                # Read min1
                wfn = self.WFN(self.ints, 'cs')
                wfn.read_from_disk(mtag)

                # Take a random step away from minimum
                step = numpy.random.rand(wfn.dim) - 0.5
                step = step / numpy.linalg.norm(step) * stepsize
                wfn.take_step(step)

                # Converge HybridEF to find a transition state
                opt = HybridEF()
                converged = opt.run(wfn, index=1, maxit=500, thresh=1e-6,plev=0)
                Ets, wfnts = wfn.energy, wfn.copy()

                # Check if we have found a transition state
                if converged and self.check_ts(wfn):
                    newts, ts_i, ts_tag = self.is_new_ts(wfn)
                    if(not newts):
                        print(f"  Transition state is equivalent to transition state {ts_i:<4d} with tag {ts_tag:s} E = {Ets: 16.10f}")
                    else:
                        print(f"  New transition state found {ts_i:4d} with tag {ts_tag:s} E = {Ets: 16.10f}")

                # If new transition state found, save it along with the connection information
                if newts:
                    # Find the connection
                    connectdata = self.connect(wfnts)
                    if(connectdata is None):
                        pass
                    else:
                        (Imin1,Emin1), (Imin2,Emin2) = connectdata
                        print(f"min{Imin1:<4d} [{Emin1: 16.10f}] <--> ts{ts_i:<4d} [{Ets: 16.10f}] <--> min{Imin2:<4d} [{Emin2: 16.10f}]")
                        self.save_ts(wfnts, Imin1, Imin2)
               
    def find_spinflip(self):
        """
        Find connections between minima via transition states using single-ended transition 
        state searches.
        """
        print("Performing spin flip on minima...")
        current_minima = dict(self.mindata)
        for _, mtag in current_minima.items():
            # Read minimum
            wfn = self.WFN(self.ints, 'cs')
            wfn.read_from_disk(mtag)

            # Perform spin-flip
            old_coeff = wfn.mo_coeff.copy()
            wfn.initialise(old_coeff[[1,0]])

            # Converge the minimum and save if new
            opt = LBFGS()
            converged = opt.run(wfn, maxit=1000, thresh=1e-6,plev=0)
            Emin, wfmin = wfn.energy, wfn.copy()
            if converged and self.check_minimum(wfn):
                newmin, min_i, min_tag = self.is_new_minimum(wfn)
                if(not newmin):
                    print(f"  Spin-flipped minimum is equivalent to minimum {min_i:<4d} with tag {min_tag:s} E = {Emin: 16.10f}")
                else:
                    print(f"  New spin-flipped minimum found {min_i:4d} with tag {min_tag:s} E = {Emin: 16.10f}")
                    self.save_minimum(wfn)

            # Converge HybridEF to find a transition state
            opt = HybridEF()
            opt.run(wfn, index=1, maxit=500, thresh=1e-6,plev=0)
            Ets, wfnts = wfn.energy, wfn.copy()

            # Check if we have found a transition state
            if self.check_ts(wfn):
                newts, ts_i, ts_tag = self.is_new_ts(wfn)
                if(not newts):
                    print(f"  Transition state is equivalent to transition state {ts_i:<4d} with tag {ts_tag:s} E = {Ets: 16.10f}")
                else:
                    print(f"  New transition state found {ts_i:4d} with tag {ts_tag:s} E = {Ets: 16.10f}")

            # If new transition state found, save it along with the connection information
            if newts:
                # Find the connection
                connectdata = self.connect(wfnts)
                if(connectdata is None):
                    pass
                else:
                    (Imin1,Emin1), (Imin2,Emin2) = connectdata
                    print(f"min{Imin1:<4d} [{Emin1: 16.10f}] <--> ts{ts_i:<4d} [{Ets: 16.10f}] <--> min{Imin2:<4d} [{Emin2: 16.10f}]")
                    self.save_ts(wfnts, Imin1, Imin2)

        print("Performing spin flip on transition states...")
        current_ts = dict(self.tsdata)
        for _, mtag in current_ts.items():
            
            # Read transition state
            wfn = self.WFN(self.ints, 'cs')
            wfn.read_from_disk(mtag)

            # Perform spin-flip
            old_coeff = wfn.mo_coeff.copy()
            wfn.initialise(old_coeff[[1,0]])

            # Converge HybridEF to find a transition state
            opt = HybridEF()
            opt.run(wfn, index=1, maxit=500, thresh=1e-6,plev=0)
            Ets, wfnts = wfn.energy, wfn.copy()

            # Check if we have found a transition state
            if self.check_ts(wfn):
                newts, ts_i, ts_tag = self.is_new_ts(wfn)
                if(not newts):
                    print(f"  Transition state is equivalent to transition state {ts_i:<4d} with tag {ts_tag:s} E = {Ets: 16.10f}")
                else:
                    print(f"  New transition state found {ts_i:4d} with tag {ts_tag:s} E = {Ets: 16.10f}")

            # If new transition state found, save it along with the connection information
            if newts:
                # Find the connection
                connectdata = self.connect(wfnts)
                if(connectdata is None):
                    pass
                else:
                    (Imin1,Emin1), (Imin2,Emin2) = connectdata
                    print(f"min{Imin1:<4d} [{Emin1: 16.10f}] <--> ts{ts_i:<4d} [{Ets: 16.10f}] <--> min{Imin2:<4d} [{Emin2: 16.10f}]")
                    self.save_ts(wfnts, Imin1, Imin2)
