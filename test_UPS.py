import numpy as np
from quantel.opt.function import Function
from quantel.opt.lbfgs import LBFGS
from quantel.opt.linear import Linear
from quantel.gnme.utils import gen_eig_sym
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper as jw
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm, expm_multiply, gmres
from timeit import default_timer as timer
import itertools


class T_UPS(Function):
    """ Tiled Unitary Product States

        Inherits from the Function abstract base class
    """
    def __init__(self,include_doubles=False, approx_prec=False, use_prec=False, use_proj=True, pp=True, oo=True, include_dmat=False):
        # Hamiltonian variables
        self.t = 1
        self.U = 6

        # define number of spin and spat orbitals
        self.no_spat = 6
        self.no_spin = self.no_spat * 2
        # Num of alpha spin
        self.no_alpha = 3
        # Num of beta spin
        self.no_beta = 3
        # Basis size of Fock Space
        self.N = 2**self.no_spin

        self.perf_pair = pp
        self.orb_opt = oo
        self.use_proj = use_proj
        self.use_prec = use_prec
        # approximate diagonal of hessian for preconditioner
        self.approx_prec = approx_prec
        if self.use_proj:
            self.initialise_projector()

        # Number of operators (parameters)
        self.include_doubles = include_doubles
        # self.nop = int(self.no_spat * (self.no_spat - 1) / 2)
        self.nop = self.no_spat - 1
        if(self.include_doubles): 
            self.nop *= 2
        
        self.initialise_tups_op_mat()
        print('Operator Matrices Generated')

        # operator order from left to right
        self.layers = 3
        self.initialise_op_order() 
        print('Operator Order Generated')

        # Define Hamiltonian and reference
        self.hamiltonian()
        print('Hamiltonian Generated')
        self.initialise_ref()
        print('Wavefunction Reference Generated')

        self.include_dmat = include_dmat
        if self.include_dmat:
            self.initialise_doubly_rm_matrix()

        # Current position
        self.x = np.zeros(self.dim)
        self.update()
        
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
        E = np.dot(np.conj(self.wfn), self.H_wfn)
        return E

    @property
    def gradient(self):
        """Get the function gradient"""
        grad = 2 * self.wfn_grad.T @ self.H_wfn
        # grad = self.wfn_grad_2
        return grad
    
    @property
    def hessian_diagonal(self):
        H_wfn_grad = self.mat_H @ self.wfn_grad
        hess1 = 2 * np.einsum('id,id->d', self.wfn_grad, H_wfn_grad)
        if not self.approx_prec: 
            # hess1 = self.wfn_hess
            hess2 = 2 * self.wfn_hess.T @ self.H_wfn
            # np.einsum('i,ij,jd->d', self.wfn, self.mat_H, self.wfn_hess, optimize=True)
            return hess1 + hess2
        else: 
            return hess1

    @property 
    def hessian(self):
        """Get the Hessian matrix of second-derivatives"""
        pass

    def get_preconditioner(self):
        if self.use_prec:
            return self.hessian_diagonal
        else:
            return np.ones(self.dim)
    
    def take_step(self,step):
        """Take a step in parameter space"""
        self.x = np.mod(self.x + step + np.pi, 2*np.pi) - np.pi
        self.update()
    
    def get_wfn(self,x):
        '''Rotates the wavefunction to generate a new wavefunction'''
        wfn = self.wf_ref.copy()
        for idx, op in enumerate(self.op_order):
            wfn = expm_multiply(self.kop_ij[op]*x[idx], wfn)
        return wfn

    def get_wfn_gradient(self,x):
        if self.use_proj:
            N = self.proj_N
        else:
            N = self.N

        if not self.approx_prec:
            wfn_grad = np.zeros((N, 2*self.dim))
        else:
            wfn_grad = np.zeros((N, self.dim))
        for j in range(wfn_grad.shape[1]):
            wfn_grad[:,j] = self.wf_ref.copy()
        
        for j, op in enumerate(self.op_order):
            wfn_grad = expm_multiply(self.kop_ij[op]*x[j], wfn_grad)
            wfn_grad[:,j] = self.kop_ij[op] @ wfn_grad[:,j]
            if not self.approx_prec:
                wfn_grad[:,j+self.dim] = self.kop_ij[op] @ wfn_grad[:,j]
        return wfn_grad[:,:self.dim], wfn_grad[:,self.dim:]

    def update(self):
        '''Updates the parameters'''
        self.wfn = self.get_wfn(self.x)
        self.H_wfn = self.mat_H @ self.wfn
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
        if self.use_proj:
            self.mat_H = csc_matrix(self.mat_proj.T @ (self.mat_H @ self.mat_proj))
    
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
        if self.perf_pair:
            strlst = [f"+_{2*i}" for i in range(self.no_alpha)] + [f"+_{2*i+self.no_spat}" for i in range(self.no_beta)]
        else:
            strlst = [f"+_{i}" for i in range(self.no_alpha)] + [f"+_{i+self.no_spat}" for i in range(self.no_beta)]

        hf_op = FermionicOp({" ".join(strlst): 1.0}, num_spin_orbitals=self.no_spin)
        mat_hf_op = jw().map(hf_op).to_matrix().real
        self.wf_ref = mat_hf_op @ wf_vac
        if self.use_proj:
            self.wf_ref = self.mat_proj.T @ self.wf_ref
    
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
        self.op_order.pop(6)
        self.op_order.pop(3)
        self.op_order.pop(0)

        oo_order = []
        for i in range(0,len(self.kop_ij),2):
            oo_order.extend([i])
        if(self.orb_opt):
            self.op_order.extend(oo_order*int(self.layers/2))

    def get_singles_matrix(self, p, q):
        t = FermionicOp({f"+_{p} -_{q}": 1.0}, num_spin_orbitals=self.no_spin)
        t += FermionicOp({f"+_{p+self.no_spat} -_{q+self.no_spat}": 1.0}, num_spin_orbitals=self.no_spin)
        k = t - t.adjoint()
        mat_k = jw().map(k).to_matrix().real
        if self.use_proj:
            mat_k = self.mat_proj.T @ (mat_k @ self.mat_proj) 
        return csc_matrix(mat_k)

    def get_doubles_matrix(self, p, q):
        t = FermionicOp({f"+_{p} +_{p+self.no_spat} -_{q+self.no_spat} -_{q}": 1.0}, num_spin_orbitals=self.no_spin)
        k = t - t.adjoint()
        mat_k = jw().map(k).to_matrix().real
        if self.use_proj:
            mat_k = self.mat_proj.T @ (mat_k @ self.mat_proj) 
        return csc_matrix(mat_k)

    def initialise_projector(self):
        # alpha spin combinations
        perm_alpha_str = '1'*self.no_alpha + '0'*(self.no_spat-self.no_alpha)
        alpha_perms = tuple(set(itertools.permutations(perm_alpha_str)))
        alpha_perms = (''.join(x) for x in alpha_perms)
        # beta spin combinations
        perm_beta_str = '1'*self.no_beta + '0'*(self.no_spat-self.no_beta)
        beta_perms = set(itertools.permutations(perm_beta_str))
        beta_perms = (''.join(x) for x in beta_perms)
        # get product of the 2 combinations
        full_perms = set(itertools.product(beta_perms,alpha_perms))
        proj_indices = []
        # convert indices from binary to decimal
        for x in full_perms:
            proj_indices.append(int(''.join(x),2))
        # sort indices from lowest to highest
        proj_indices.sort()

        # construct projector matrix and dimension of reduced space
        self.mat_proj = np.zeros((self.N, len(proj_indices)))
        for j, idx in enumerate(proj_indices):
            self.mat_proj[idx][j] = 1
        self.mat_proj = csc_matrix(self.mat_proj)
        self.proj_N = len(proj_indices)
    
    def initialise_doubly_rm_matrix(self):
        self.doubly_rm_mat = np.zeros((self.no_spat,self.no_spat, self.N, self.N))
        for r in range(self.no_spat):
            for s in range(self.no_spat):
                op = FermionicOp({f"-_{s+self.no_spat} -_{r}": 1.0}, num_spin_orbitals=self.no_spin)
                op += FermionicOp({f"-_{s} -_{r+self.no_spat}": 1.0}, num_spin_orbitals=self.no_spin)
                op += FermionicOp({f"-_{s+self.no_spat} -_{r+self.no_spat}": 1.0}, num_spin_orbitals=self.no_spin)
                op += FermionicOp({f"-_{s} -_{r}": 1.0}, num_spin_orbitals=self.no_spin)
                mat_op = jw().map(op).to_matrix().real
                self.doubly_rm_mat[r,s,:,:] = mat_op

    def get_2e_red_density_mat(self):
        ket = self.mat_proj @ self.wfn 
        ket = self.doubly_rm_mat @ ket
        density_mat = np.einsum('pqi, rsi->pqrs', ket, ket)
        return density_mat


np.random.seed(10)
trials = 1
data = np.zeros((1,2))
opt = LBFGS(with_transport=False,with_canonical=False,prec_thresh=0.1)
lin = Linear()
for isample in range(trials):
    test = T_UPS(include_doubles=True, approx_prec=True, use_prec=True, pp=True, oo=True)
    # test.get_initial_guess()
    # print('Initial Guess Applied')
    print(f"Use preconditioner: {test.use_prec}")
    print(f"Approximate preconditioner: {test.approx_prec}")
    print(f"Orbital Optimised: {test.orb_opt}")
    print(f"Perfect Pairing: {test.perf_pair}")
    lin.run_dogleg(test,maxit=1000)
    print(test.x)
    # iterations, energy = lin.run_linesearch(test, maxit=1000)
    # data[isample,:] = iterations, energy
    # with open("./dump/random/l-bfgs.csv", "ab") as f:
    #     np.savetxt(f, data, delimiter=",")