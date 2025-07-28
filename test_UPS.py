import numpy as np
from quantel.opt.function import Function
from quantel.opt.lbfgs import LBFGS
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper as jw
from scipy.linalg import expm

class T_UPS(Function):
    """ Tiled Unitary Product States

        Inherits from the Function abstract base class
    """
    def __init__(self,include_doubles=False):
        self.t = 1
        self.U = 6
        self.no_spat = 4
        self.no_spin = self.no_spat * 2
        # Num of alpha spin
        self.no_alpha = 2           
        # Num of alpha spin
        self.no_beta = 2
        # Basis size of Fock Space
        self.N = 2**self.no_spin

        # operator order from left to right
        self.op_order = [0,1,0,0,1,0]     

        # Number of operators (parameters)
        self.include_doubles = include_doubles
        self.nop = int(self.no_spat * (self.no_spat - 1) / 2)
        if(self.include_doubles): self.nop *= 2
        self.initialise_op_mat()

        # Define Hamiltonian and reference
        self.hamiltonian()
        self.initialise_ref()

        # Current position
        self.x = np.zeros(self.dim)
        self.update()

    @property
    def dim(self):
        """Dimension of parameter matrix"""
        return len(self.op_order)

    @property
    def value(self):
        """Get the corresponding variational value"""
        return self.energy

    @property
    def energy(self):
        E = np.dot(np.conj(self.wfn), self.mat_H @ self.wfn)
        return E

    @property
    def gradient(self):
        """Get the function gradient"""
        return 2* self.wfn_grad @ (self.mat_H @ self.wfn)
    
    @property
    def hessian_diagonal(self):
        hess1 = 2 * np.einsum('i,ij,dj->d', self.wfn, self.mat_H, self.wfn_hess)
        hess2 = 2 * np.einsum('di,ij,dj->d', self.wfn_grad, self.mat_H, self.wfn_grad)
        return hess1 + hess2
        hess_diag = np.zeros(self.dim)



        wfn_grad_sq = np.zeros((self.dim, self.N))
        for j in range(self.dim):
            tmp = self.wf_ref
            # transform reference until the jth parameter
            for idx, op in enumerate(self.op_order[:j]):
                tmp = expm(self.kop_ij[op]*self.x[idx]) @ tmp
            # multiply with operator matrix
            tmp = self.kop_ij[self.op_order[j]] @ tmp
            tmp = self.kop_ij[self.op_order[j]] @ tmp
            # continue transforming the reference wavefunction
            for idx, op in enumerate(self.op_order[j:]):
                tmp = expm(self.kop_ij[op]*self.x[idx+j]) @ tmp
            wfn_grad_sq[j] = tmp
        
        #hess1 = 2*wfn_grad_sq @ (self.mat_H @ self.wfn)
        #hess2 = np.diag(2*self.wfn_gradient @ (self.mat_H @ self.wfn_gradient.T))

        tmp = np.einsum('ij, dj->di', self.mat_H, wfn_grad_sq)
        hess_diag += 2*np.einsum('i,di->d', self.wfn, tmp)

        tmp = np.einsum('ij, dj->di', self.mat_H, self.wfn_gradient)
        hess_diag += 2*np.einsum('di,di->d', self.wfn_gradient, tmp)
        return hess_diag


    @property    
    def hessian(self):
        """Get the Hessian matrix of second-derivatives"""
        pass

    def get_preconditioner(self):

        return np.ones(self.dim)# * self.hessian_diagonal
    
    def take_step(self,step):
        """Take a step in parameter space"""
        self.x = self.x + step
        self.update()
    
    def get_wfn(self,x):
        '''Rotates the wavefunction to generate a new wavefunction'''
        U = np.identity(self.N)
        wfn = self.wf_ref.copy()
        # K = np.einsum('pij,p->ij',self.kop_ij,self.x)
        for idx, op in enumerate(self.op_order):
            wfn = expm(self.kop_ij[op]*x[idx]) @ wfn
        return wfn

    def get_wfn_gradient(self,x):
        wfn_grad = np.zeros((2*self.dim, self.N))
        for j in range(2*self.dim):
            wfn_grad[j] = self.wf_ref.copy()

        for j, op in enumerate(self.op_order):
            U = expm(self.kop_ij[op]*x[j])
            wfn_grad = np.einsum('pq,jq->jp', U, wfn_grad)
            wfn_grad[j] = self.kop_ij[self.op_order[j]] @ wfn_grad[j]
            wfn_grad[j+self.dim] = self.kop_ij[self.op_order[j]] @ wfn_grad[j]

        return wfn_grad[:self.dim], wfn_grad[self.dim:]    
    
    def update(self):
        '''Updates the parameters'''
        self.wfn = self.get_wfn(self.x)
        self.wfn_grad, self.wfn_hess = self.get_wfn_gradient(self.x)
        # TODO: You might store current wfn so you have to keep evaluating 
        #       for computing gradients and/or energy 
        pass
    
    def save_last_step(self):
        """Save current position"""
        self.xsave = self.x.copy()

    def restore_last_step(self):
        """Return wavefunction to previous position"""
        self.x = self.xsave.copy()
        self.update()

    def hamiltonian(self):
        '''Initialises the Hamiltonian matrix'''
        H = FermionicOp({'':0},num_spin_orbitals=self.no_spin)
        # one body alpha
        for p in range(self.no_spat-1):
            q = p+1
            H += -self.t*FermionicOp({f'+_{p} -_{q}':1, f'+_{q} -_{p}':1}, num_spin_orbitals=self.no_spin)

        # one body beta
        for p in range(self.no_spat, self.no_spin-1):
            q = p+1
            H += -self.t*FermionicOp({f'+_{p} -_{q}':1, f'+_{q} -_{p}':1}, num_spin_orbitals=self.no_spin)

        # two body
        for q in range(self.no_spat):
            p = q + self.no_spat
            H += FermionicOp({f"+_{q} +_{p} -_{p} -_{q}": self.U}, num_spin_orbitals=self.no_spin)
        
        # Save as a matrix
        self.mat_H = jw().map(H).to_matrix().real
    
    def get_initial_guess(self):
        # Generate current position
        self.x = 2*np.pi*(np.random.rand(self.dim)-0.5)
        self.update()

    def initialise_ref(self):
        '''Initialises the HF reference state'''
        # create vacuum state
        wf_vac = np.zeros((self.N))
        wf_vac[0] = 1

        # create HF reference state
        strlst = [f"+_{i}" for i in range(self.no_alpha)] + [f"+_{i+self.no_spat}" for i in range(self.no_beta)]
        hf_op = FermionicOp({" ".join(strlst): 1.0}, num_spin_orbitals=self.no_spin)
        mat_hf_op = jw().map(hf_op).to_matrix().real
        self.wf_ref = mat_hf_op @ wf_vac
    
    def initialise_op_mat(self):
        '''Initialise matrices for the 2nd quantised operators'''
        self.kop_ij = np.zeros((self.nop,self.N,self.N))
        # paired single
        count = 0
        for p in range(self.no_spat):
            for q in range(p):
                t = FermionicOp({f"+_{p} -_{q}": 1.0}, num_spin_orbitals=self.no_spin)
                t += FermionicOp({f"+_{p+self.no_spat} -_{q+self.no_spat}": 1.0}, num_spin_orbitals=self.no_spin)
                k = t - t.adjoint()
                mat_k = jw().map(k).to_matrix().real
                self.kop_ij[count,:,:] = mat_k
                count += 1
        # paired doubles
        if(self.include_doubles):
            for p in range(self.no_spat):
                for q in range(p):
                    t = FermionicOp({f"+_{p} +_{p+self.no_spat} -_{q+self.no_spat} -_{q}": 1.0}, num_spin_orbitals=self.no_spin)
                    k = t - t.adjoint()
                    mat_k = jw().map(k).to_matrix().real
                    self.kop_ij[count,:,:] = mat_k
                    count += 1

np.random.seed(7)
test = T_UPS(include_doubles=True)
test.get_initial_guess()

print(test.x)
print(test.gradient)
print(test.get_numerical_gradient())
#hess = test.get_numerical_hessian()
#hess_an = test.hessian_diagonal
#print(np.diag(hess))
#print(hess_an)

for isample in range(1):
    test = T_UPS(include_doubles=True)
    
    test.get_initial_guess()
    opt = LBFGS(with_transport=False,with_canonical=False)
    opt.run(test)
quit()