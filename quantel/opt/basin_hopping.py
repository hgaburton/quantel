#!/usr/bin/python3

import copy
import sys
import numpy as np
from quantel.opt.lbfgs import LBFGS


class BasinHopping():
    '''
        Class to implement and control Basin Hopping procedure for global optimization.
    '''
    def __init__(self, optimizer=LBFGS(), **kwargs):
        self.optimizer = optimizer
        self.control = dict()
        # Temperature for the Metropolis acceptance criterion
        self.control["temperature"] = kwargs.get("temperature", 1.0)
        # Maximum step size for the random perturbation of the state
        self.control["stepsize"] = kwargs.get("stepsize", 0.5)
        # Number of minima to store
        self.control["nminima"] = kwargs.get("nminima", 10)
        # Target acceptance rate for the Metropolis criterion
        self.control["target_accept"] = kwargs.get("target_accept", 0.5)
        # How often accept/reject is checked
        self.control["change_accept"] = kwargs.get("change_accept", 50)
        # Maximum number of iterations for the L-BFGS optimization
        self.control["opt_maxit"] = kwargs.get("opt_maxit", 1000)
        # Loose convergence threshold for BH steps
        self.control["loose_convergence"] = kwargs.get("loose_convergence", 1e-4)
        # Tight convergence threshold for final optimization of minima
        self.control["tight_convergence"] = kwargs.get("tight_convergence", 1e-7)
        # Energy tolerance for considering two minima as distinct
        self.control["distinct_etol"] = kwargs.get("distinct_etol", 1e-4)        
        # Overlap tolerance for considering two minima as distinct (if applicable)
        self.control["distinct_stol"] = kwargs.get("distinct_stol", 1e-3)
        # Control if overlap used to compare distinct minima (can be expensive for CSFs)
        self.control["check_overlap"] = kwargs.get("check_overlap", False)
        # Seed for random number generator
        self.control["random_seed"] = kwargs.get("random_seed", None)
        if self.control["random_seed"] is not None:
            np.random.seed(self.control["random_seed"])
        # Kwargs for optimizer
        self.control["opt_kwargs"] = kwargs.get("opt_kwargs", dict(plev=0, maxit=500))
        self.control["opt_kwargs"]["thresh"] = self.control["loose_convergence"]
        # Print level for logging
        self.plev = kwargs.get("plev", 1)

        # Reset all internal storage.
        self.reset()


    def reset(self):
        '''
            Reset the internal state of the Basin Hopping procedure.
        '''
        # List of (energy, state) tuples for all minima found
        self.all_solutions = []
        # Initial temperature
        self.temperature = self.control["temperature"]
        self.stepsize = self.control["stepsize"]
        # Current acceptance rate
        self.naccept = 0
        self.nreject = 0


    def new_solution(self, state):
        ''' 
            Check if a minumum is distinct from those already found.
        '''
        energy = state.energy
        # First solution, must be new
        if not self.all_solutions:
            if self.plev > 0: 
                print(f"  New solution with energy = {energy: .10f} Eh")
            return True
        
        # Check energy (and optionally state overlap) against all solutions already found
        energy_check, overlap_check = True, True
        for ej, statej in self.all_solutions:
            if(abs(ej - energy) < self.control["distinct_etol"]):
                # Optionally check overlap (gets expensive for CSFs)
                energy_check = False
                # TODO: Implement a density-based overlap check for CSFs to avoid expensive wavefunction overlaps
                if(self.control["check_overlap"]):
                    if(abs(1-abs(state.overlap(statej))) < self.control["distinct_stol"]): 
                        overlap_check = False
                        return False
                else:
                    return False
                
        # Made it to the end, so must be new
        if self.plev > 0: 
            print(f"  New solution with energy = {energy: .10f} Eh")
        return True


    def save_solution(self, state):
        ''' 
            Check whether a solution is distinct from those already found, and if so save it in the list of solutions.
        '''
        # Append solution to list of solutions and sort by energy
        self.all_solutions.append((state.energy, state.copy()))
        self.all_solutions.sort(key=lambda x: x[0])

        # Truncate the list of solutions to the maximum number allowed
        if(len(self.all_solutions) > self.control["nminima"]):
            self.all_solutions = self.all_solutions[:self.control["nminima"]]
        

    def update_acceptreject(self):
        ''' 
            Update the step-size based on the current acceptance rate and the target acceptance rate.
        '''
        ntrial = self.naccept + self.nreject
        if ntrial == 0:
            return
        accept_rate = self.naccept / ntrial
        # Simple proportional control to adjust step-size
        if accept_rate > self.control["target_accept"]:
            self.stepsize *= 1.05
        else:
            self.stepsize /= 1.05
        if self.plev > 1:
            print(f" Acceptance rate for last {ntrial} steps: {accept_rate:.3f}   Updated step-size: {self.stepsize:.4e}")
        
        # Reset counters
        self.naccept = 0
        self.nreject  = 0


    def run(self, wfn, nhop=100):
        '''
            Run the Basin Hopping optimization starting from the given initial state.

            Parameters
            ----------
            wfn  : object
                The initial state from which to start the optimization. This should be compatible with the optimizer used.
            nhop : int
                The number of Basin Hopping steps to perform.
        '''
        # Print a summary of the algorithm parameters
        if self.plev > 0: 
            print()
            print("  Initializing series of Basin-Hopping steps...")
            print(f"    > Number of BH steps  = {nhop}")
            print(f"    > Random seed         = {self.control['random_seed']}")
            print(f"    > Temperature         = {self.temperature:.4e}")
            print(f"    > Initial step size   = {self.stepsize:.4e}")
            print(f"    > Target acceptance   = {self.control['target_accept']:.3f}")
            print(f"    > Check accept every  = {self.control['change_accept']} steps")
            print(f"    > Energy tolerance    = {self.control['distinct_etol']:.3e} Eh")
            if self.control["check_overlap"]:
                print(f"    > Overlap tolerance   = {self.control['distinct_stol']:.3e}")
            print(f"    > Optimizer           = {self.optimizer.__class__.__name__}")
            print(f"    > Optimizer maxit     = {self.control['opt_kwargs']['maxit']}")
            print(f"    > Loose convergence   = {self.control['loose_convergence']:.3e} Eh")
            print() 

        # Run the initial optimization to find the first solution
        converged = self.optimizer.run(wfn, **self.control["opt_kwargs"])
        if self.plev > 0:
            print(f"  Hop   0 (initial):  E = {wfn.energy: .10f}  T_hop={self.temperature:.4e}")
            sys.stdout.flush()
        # Save initial state as best solution found in this run so far
        wfn_best = wfn.copy()
        # Save the first solution if it is distinct
        if(self.new_solution(wfn)): 
            self.save_solution(wfn)

        # Main Basin Hopping loop
        for hop in range(nhop):
            # Generate a new trial state by randomly perturbing the current state
            trial_wfn = wfn.copy()
            # Take a random step 
            step = self.stepsize * (2 * np.random.rand(trial_wfn.dim) - 1)
            #step *= self.stepsize / np.linalg.norm(step)
            trial_wfn.take_step(step)
           
            # Optimize the trial state to find the local solution
            converged = self.optimizer.run(trial_wfn,  **self.control["opt_kwargs"])
            # Save new best solution if it is the lowest found in this run so far
            if (trial_wfn.energy < wfn_best.energy + 1e-12): 
                wfn_best = trial_wfn.copy()

            # Decide whether to accept the new solution based on the Metropolis criterion
            delta_e = trial_wfn.energy - wfn.energy
            if delta_e < 0.0:
                accepted = True
                comment  = f"downhill  dE={delta_e:+.3e}"
            elif self.temperature > 0.0:
                prob     = np.exp(-delta_e / self.temperature)
                accepted = np.random.rand() < prob
                comment  = f"{'accepted' if accepted else 'rejected'} Metropolis  dE={delta_e:+.3e}  Prob={prob:.3f}"
            else:
                accepted = False
                comment  = f"rejected  dE={delta_e:+.3e}"
            
            # If accepted, update the current state to the new solution
            if accepted:
                wfn = trial_wfn.copy()
                self.naccept += 1
            else:
                self.nreject += 1

            # Update acceptance rate and adjust step-size if necessary
            if((hop+1) % self.control["change_accept"] == 0):
                self.update_acceptreject()
            
            # Print progress
            if self.plev > 0:
                print(f"  Hop {hop+1:6d}:  E = {trial_wfn.energy: .10f}  E_markov = {wfn.energy: .10f}  {comment}")
                sys.stdout.flush()
        
            # Check if the new solution is distinct and save it if so
            if(self.new_solution(trial_wfn)): 
                self.save_solution(trial_wfn)

        # Set wfn to the best solution found in this run
        wfn = wfn_best.copy()
        return 
    

    def final_quench(self):
        ''' 
            Perform a final quench of all solutions found to ensure they are fully converged.
        '''
        test_minima = [state.copy() for (_,state) in self.all_solutions]
        self.all_solutions = []
        if self.plev > 0:
            print("\nPerforming final quench of all saved solutions...")
        for idx, state in enumerate(test_minima):
            converged = self.optimizer.run(state, **{**self.control["opt_kwargs"], "thresh": self.control["tight_convergence"]})
            if self.plev > 0:
                print(f" Final quench of solution {idx+1: 5d}", end="")
            if(self.new_solution(state)): 
                self.save_solution(state)
        
        return self.all_solutions