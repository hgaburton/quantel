#!/usr/bin/python3

from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.linalg

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


    def get_numerical_hessian(self,eps=1e-3):
        """Finite difference Hessian matrix for debugging"""
        Hess = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            print(i, self.dim)
            for j in range(i,self.dim):
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
