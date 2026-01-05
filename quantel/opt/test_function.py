#!/usr/bin/python3

from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.linalg
from quantel.opt.davidson import Davidson

class Function(metaclass=ABCMeta):
    """Abstract base class for a real-valued objective function"""
    @property
    @abstractmethod
    def dim(self):
        """Number of variables"""
        pass

    @property
    @abstractmethod
    def value(self):
        """Get the corresponding variational value
            
        Returns:
            The corresponding variational value.
        """
        pass

    @property
    @abstractmethod
    def gradient(self):
        """Get the function gradient"""
        pass

    @property
    @abstractmethod
    def hessian(self):
        """Get the Hessian matrix of second-derivatives"""
        pass

    @abstractmethod
    def take_step(self,step):
        """Take a step in parameter space"""
        pass

    @abstractmethod
    def save_last_step(self):
        """Save current position"""
        pass

    @abstractmethod
    def restore_last_step(self):
        """Return wavefunction to previous position"""
        pass


    def get_numerical_gradient(self,eps=1e-3):
        """Finite difference gradient for debugging"""
        grad = np.zeros((self.dim))
        self.save_last_step()
        for i in range(self.dim):
            x1 = np.zeros(self.dim)
            x2 = np.zeros(self.dim)
                
            x1[i] += eps
            x2[i] -= eps
                
            self.take_step(x1)
            f1 = self.value
            self.restore_last_step()

            self.take_step(x2)
            f2 = self.value
            self.restore_last_step()

            grad[i] = (f1 - f2) / (2 * eps)

        return grad


    def get_numerical_hessian(self,eps=1e-3,diag=False):
        """Finite difference Hessian matrix for debugging"""
        Hess = np.zeros((self.dim, self.dim))
        # Save the origin
        self.save_last_step()

        # Compute finite differences
        for i in range(self.dim):
            for j in range(i,self.dim):
                if(diag and i!=j): continue
                x1 = np.zeros(self.dim)
                x2 = np.zeros(self.dim)
                x3 = np.zeros(self.dim)
                x4 = np.zeros(self.dim)
                
                x1[i] += eps; x1[j] += eps
                x2[i] += eps; x2[j] -= eps
                x3[i] -= eps; x3[j] += eps
                x4[i] -= eps; x4[j] -= eps
                
                self.take_step(x1)
                f1 = self.value
                self.restore_last_step()

                self.take_step(x2)
                f2 = self.value
                self.restore_last_step()

                self.take_step(x3)
                f3 = self.value
                self.restore_last_step()

                self.take_step(x4)
                f4 = self.value
                self.restore_last_step()

                Hess[i,j] = ((f1 - f2) - (f3 - f4)) / (4 * eps * eps)
                if(i!=j): Hess[j,i] = Hess[i,j]

        return Hess


    def get_hessian_index(self, tol=1e-16):
        """Compute the Hessian index
               tol : Threshold for determining a zero eigenvalue
        """
        eigs = scipy.linalg.eigvalsh(self.hessian)
        ndown = 0
        nzero = 0
        nuphl = 0
        for i in eigs:
            if i < -tol:  ndown += 1
            elif i > tol: nuphl +=1
            else:         nzero +=1 
        return (ndown, nzero, nuphl)

    def get_davidson_hessian_index(self, ntarget=5, eps=1e-5):
        """Iteratively compute Hessian index from gradient only. 
           This approach uses the Davidson algorithm."""
        # Get approximate diagonal terms
        diag = self.get_preconditioner() # the diagonal values which approximate the full Hessian matrix,this is only a 1D array

        # Start with 5 eigenvalues
        nv = ntarget
        #initialise class instance with max size nreset
        david = Davidson(nreset=50)
        # Initialise from the lowest diagonal elements
        x = np.zeros((diag.size, nv),order='F') #order just tells how to store the matrix in memory, i.e. in row-major style 'C' or column major 'F'
        for i, j in enumerate(np.argsort(diag)[:nv]):
            #np.argsort(a) returns a list rearranged indices ordering the diag elements from the smallest to largest
            #np.argsort(a)[:nv] then limits it to the most negative nv elements
            #enumerate(np.argsort()), i runs normally from 0 -> nv, and j is corresponding index for the ith most negative diag element 
            x[j,i] = 1.0 #this places a single 1.0 in the row vector corresponding to the correct index of diag. 
        #the matrix x is hence an initial guess for the eigenvectors of the Hessian   
        while True:
            # Get lowest eigenvalues through Davidson algorithm
            eigs, x = david.run(self.approx_hess_on_vec,diag,nv,
                                xguess=x,plev=2,tol=1e-4,maxit=1000,Hv_args=dict(eps=1e-5))
            # gives our subspace projector x, and our eigenvalues
            if(np.any(eigs > 0)):
                # We have found the first positive eigenvalue, so we can break, as we only care about finding the total number of negative indices
                break

            # Augment with more columns and try again
            x  = np.column_stack([x, np.random.rand(diag.size,5)]) #adding 5 random vectors onto subspace vectors
            nv = x.shape[1] #update the size
 
        # Count the Hessian index
        ndown = 0
        nzero = 0
        for i in eigs:
            if i < -1e-16:  ndown += 1
            elif not i>1e-16:  nzero +=1 

        # Save the result
        self.hess_index = (ndown, nzero)
        return


    def pushoff(self, n, angle=np.pi/2):
        """Perturb function along n Hessian eigenvector directions"""
        eigval, eigvec = np.linalg.eigh(self.hessian)
        step = sum(eigvec[:,i] * angle for i in range(n))
        self.take_step(step)


    def check_gradient(self, tol=1e-3):
        anl = self.gradient
        num = self.get_numerical_gradient()
        diff = anl - num
        print(diff)
        return np.linalg.norm(diff) / diff.size < tol

    def check_hessian(self, tol=1e-3):
        anl = self.hessian
        num = self.get_numerical_hessian()
        diff = anl - num
        print(diff)
        return np.linalg.norm(diff) / diff.size < tol
