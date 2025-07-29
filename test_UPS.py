import numpy as np
from quantel.opt.function import Function
from quantel.opt.lbfgs import LBFGS
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper as jw
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm, expm_multiply
from timeit import default_timer as timer


class T_UPS(Function):
    """ Tiled Unitary Product States

        Inherits from the Function abstract base class
    """
    def __init__(self,include_doubles=False, approx_prec=False):
        # Hamiltonian variables
        self.t = 1
        self.U = 6

        # define number of spin and spat orbitals
        self.no_spat = 6
        self.no_spin = self.no_spat * 2
        # Num of alpha spin
        self.no_alpha = 3
        # Num of alpha spin
        self.no_beta = 3
        # Basis size of Fock Space
        self.N = 2**self.no_spin

        # approximate diagonal of hessian for preconditioner
        self.approx_prec = approx_prec

        # Number of operators (parameters)
        self.include_doubles = include_doubles
        # self.nop = int(self.no_spat * (self.no_spat - 1) / 2)
        self.nop = self.no_spat - 1
        if(self.include_doubles): 
            self.nop *= 2
        
        self.initialise_tups_op_mat()
        print('Operator Matrices Generated')

        # operator order from left to right
        self.layers = 6
        self.initialise_op_order() 
        print('Operator Order Generated')

        
        # Define Hamiltonian and reference
        self.hamiltonian()
        print('Hamiltonian Generated')
        self.initialise_ref()
        print('Wavefunction Reference Generated')

        # Current position
        # self.x = np.zeros(self.dim)
        # self.update()

    @property
    def dim(self):
        """Dimension of parameter vector, x"""
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
        return 2* self.wfn_grad.T @ (self.mat_H @ self.wfn)
    
    @property
    def hessian_diagonal(self):
        hess2 = 2 * np.einsum('id,ij,jd->d', self.wfn_grad, self.mat_H, self.wfn_grad)
        if self.approx_prec == False: 
            hess1 = 2 * np.einsum('i,ij,jd->d', self.wfn, self.mat_H, self.wfn_hess)
            return hess1 + hess2
        else: 
            return hess2

    @property 
    def hessian(self):
        """Get the Hessian matrix of second-derivatives"""
        pass

    def get_preconditioner(self):

        return np.ones(self.dim) * self.hessian_diagonal
    
    def take_step(self,step):
        """Take a step in parameter space"""
        self.x = self.x + step
        self.update()
    
    def get_wfn(self,x):
        '''Rotates the wavefunction to generate a new wavefunction'''
        wfn = self.wf_ref.copy()
        for idx, op in enumerate(self.op_order):
            wfn = expm_multiply(self.kop_ij[op]*x[idx], wfn)
        return wfn

    def get_wfn_gradient(self,x):
        if self.approx_prec == False:
            wfn_grad = np.zeros((self.N, 2*self.dim))
        else:
            wfn_grad = np.zeros((self.N, self.dim))

        for j in range(wfn_grad.shape[1]):
            wfn_grad[:,j] = self.wf_ref.copy()

        for j, op in enumerate(self.op_order):
            wfn_grad = expm_multiply(self.kop_ij[op]*x[j], wfn_grad)
            wfn_grad[:,j] = self.kop_ij[self.op_order[j]] @ wfn_grad[:,j]
            if self.approx_prec == False:
                wfn_grad[:,j+self.dim] = self.kop_ij[self.op_order[j]] @ wfn_grad[:,j]

        return wfn_grad[:,:self.dim], wfn_grad[:,self.dim:]    
    
    def update(self):
        '''Updates the parameters'''
        self.wfn = self.get_wfn(self.x)
        self.wfn_grad, self.wfn_hess = self.get_wfn_gradient(self.x)


    def save_last_step(self):
        """Save current position"""
        self.xsave = self.x.copy()

    def restore_last_step(self):
        """Return wavefunction to previous position"""
        self.x = self.xsave.copy()
        self.update()

    def hamiltonian(self):
        '''Initialises the Hamiltonian matrix for the Hubbard system'''
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
        # Generate initial position
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
        '''Initialise matrices using the 2nd quantised operators - general case'''
        self.kop_ij = {}
        # paired single
        count = 0
        for p in range(self.no_spat):
            for q in range(p):
                self.kop_ij[count] = self.get_singles_matrix(p,q)
                count += 1
        # paired doubles
        if(self.include_doubles):
            for p in range(self.no_spat):
                for q in range(p):
                    self.kop_ij[count] = self.get_doubles_matrix
                    count += 1
    
    def initialise_tups_op_mat(self):
        '''Initialise matrices using the 2nd quantised operators - tUPS case'''
        self.kop_ij = {}
        # paired single
        count = 0
        # defining k_10, k_32, k_54, ... k_pq. where q is even 
        for p in range(1, self.no_spat, 2):
            q = p-1
            self.kop_ij[count] = self.get_singles_matrix(p,q)
            count += 1

            # paired doubles
            if(self.include_doubles):
                self.kop_ij[count] = self.get_doubles_matrix(p,q)
                count += 1
        # defining k_21, k_43, k_65, ... k_pq. where q is odd 
        for q in range(1, self.no_spat-1, 2):
            p = q+1
            self.kop_ij[count] = self.get_singles_matrix(p,q)
            count += 1

            # paired doubles
            if(self.include_doubles):
                self.kop_ij[count] = self.get_doubles_matrix(p,q)
                count += 1
        
    def initialise_op_order(self):
        self.op_order = []
        for i in range(0,len(self.kop_ij),2):
            self.op_order.extend([i,i+1,i])
        self.op_order.extend(self.op_order*(self.layers-1))

    def get_singles_matrix(self, p, q):
        t = FermionicOp({f"+_{p} -_{q}": 1.0}, num_spin_orbitals=self.no_spin)
        t += FermionicOp({f"+_{p+self.no_spat} -_{q+self.no_spat}": 1.0}, num_spin_orbitals=self.no_spin)
        k = t - t.adjoint()
        mat_k = jw().map(k).to_matrix().real
        return csc_matrix(mat_k)

    def get_doubles_matrix(self, p, q):
        t = FermionicOp({f"+_{p} +_{p+self.no_spat} -_{q+self.no_spat} -_{q}": 1.0}, num_spin_orbitals=self.no_spin)
        k = t - t.adjoint()
        mat_k = jw().map(k).to_matrix().real
        return csc_matrix(mat_k)



np.random.seed(7)
# test = T_UPS(include_doubles=True, approx_prec=False)
# test.get_initial_guess()

# # print(test.x)
# print(test.gradient)
# print(test.get_numerical_gradient())
# hess = test.get_numerical_hessian()
# hess_an = test.hessian_diagonal
# print(np.diag(hess))
# print(hess_an)
# quit()
for isample in range(1):
    test = T_UPS(include_doubles=True, approx_prec=False)
    test.get_initial_guess()
    print('Initial Guess Applied')
    opt = LBFGS(with_transport=False,with_canonical=False)
    opt.run(test, maxit=500)