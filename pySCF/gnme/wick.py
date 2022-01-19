#!/usr/bin/python

import unittest
import numpy as np
from .utils import lowdin_pair, reduced_overlap, orthogonalise
from pyscf.fci import cistring
from itertools import permutations

def bin2(b,n):
    return '0b'+bin(b)[2:].rjust(n,'0')

def _occlst(b,n):
    return np.array([int(x) for x in bin(b)[2:].rjust(n,'0')])

class gnme_pair:

    def __init__(self, Cx, Cw, ncore, ne, nactive, S):
        # Store number of core and active orbitals
        self.ncore   = ncore
        self.nactive = nactive
        self.ne      = ne
        self.nbsf    = S.shape[0]
        self.ref     = cistring.make_strings(range(nactive), ne-ncore)[0]

        # Check everything is sensible
        assert(Cx.shape[0]==self.nbsf)
        assert(Cw.shape[0]==self.nbsf)

        # Store copy of active orbitals
        self.Cx = np.copy(Cx[:,self.ncore:(self.ncore+self.nactive)])
        self.Cw = np.copy(Cw[:,self.ncore:(self.ncore+self.nactive)])

        # Lowdin pair orbitals and compute reduced overlap
        Cx_p, Cw_p, Sxx = lowdin_pair(np.copy(Cx)[:,:self.ne], np.copy(Cw)[:,:self.ne], S)
        invSxx, self.redOV, zeros, self.nz = reduced_overlap(Sxx) 

        # Construct co-density matrices
        self.wxM = np.zeros((2,self.nbsf,self.nbsf))
        # M matrix
        self.wxM[0,:,:] = np.einsum('ip,pq,jq->ij',Cw_p,np.diag(invSxx),np.conj(Cx_p))
        for ind in zeros:
            self.wxM[0,:,:] += Cx_p[:,ind].dot(np.conj(Cx_p[:,ind].T))
        # P matrix
        for ind in zeros:
            self.wxM[1,:,:] += Cw_p[:,ind].dot(np.conj(Cx_p[:,ind].T))

        # Initialise X contractions
        Cstack = np.hstack((self.Cx,self.Cw))
        self.X = np.einsum('mp,mn,ins,st,tq->ipq',np.conj(Cstack),S,self.wxM,S,Cstack)

        # Initialise Y contractions
        self.Y = self.X.copy()
        self.Y[0,:,:] -= np.einsum('mp,mn,nq->pq',np.conj(Cstack),S,Cstack)

    def get_excitation_info(self, bx, bw):
        occref = _occlst(self.ref,self.nactive)
        occx   = _occlst(bx,self.nactive)
        occw   = _occlst(bw,self.nactive)

        exx    = occx - occref
        hpx    = (self.nactive-1)-np.hstack((np.argwhere(exx==-1),np.flip(np.argwhere(exx==1))))

        exw    = occw - occref
        hpw    = (self.nactive-1)-np.hstack((np.argwhere(exw==-1),np.flip(np.argwhere(exw==1))))

        return hpx, hpw

    def get_overlap(self, bx, bw):

        # Zero the output
        S = 0.0

        # Determine the excitations
        xhp, whp = self.get_excitation_info(bx,bw)
        nw, nx = whp.shape[0], xhp.shape[0]
        whp    += self.nactive

        # There is no overlap if fewer excitations than zero-overlap orbitals 
        if(nx + nw < self.nz): return 0.0

        # Work through different cases
        if(nx == 0 and nw == 0): 
            return self.redOV, (1.0 if (self.nz == 0) else 0.0)
        elif(nx == 0 ^ nw == 0):
            rows = xhp[:,[1]] if (nx > 0) else whp[:,[0]]
            cols = xhp[:,[0]] if (nx > 0) else whp[:,[1]]
        else:
            rows = np.vstack((xhp[:,[1]],whp[:,[0]]))
            cols = np.vstack((xhp[:,[0]],whp[:,[1]]))

        # Flatten
        rows = rows.flatten()
        cols = cols.flatten()

        if(nx + nw == 1):
            return self.redOV, self.X[self.nz,rows[0],cols[0]]
        else:
            # Combine the upper/lower triangular matrices
            idx = np.ix_(rows,cols)
            D    = np.tril(self.X[0,:,:][idx]) + np.triu(self.Y[0,:,:][idx],k=1)
            Dbar = np.tril(self.X[1,:,:][idx]) + np.triu(self.Y[1,:,:][idx],k=1)

            # Distribute nz zeros aomg columns of D
            # Corresponds to inserting nz columns of Dbar into D for every permutation
            mask = np.zeros(nx+nw,int)
            mask[:self.nz] = 1
            for perm in set(permutations(mask)):
                ind = np.array(perm)
                S += np.linalg.det(D.dot(np.diag(1-ind)) + Dbar.dot(np.diag(ind)))

        return self.redOV, S



class test_utils(unittest.TestCase):
    def get_random_coeff(self,nb):
        return np.random.rand(nb,nb)

    def get_random_overlap(self,nb):
        # Fill with random numbers
        S  = np.random.rand(nb,nb)
        # Make sure it's symmetric
        S  = 0.5 * (S + S.T)
        S  = S - np.diag(np.diag(S)) + np.eye(nb)
        # Make sure it's positive definite
        sig, U = np.linalg.eigh(S)
        S  = np.einsum('pq,qr,rs->ps',U,np.diag(np.abs(sig)),U.T)
        return S

    def test_lowdin_pair(self):

        from scipy.linalg import expm

        # Set some number of electrons (ne) and basis functions (nb)
        nc,ne,nb = 2, 4, 8

        # Get random coefficients
        Cx = self.get_random_coeff(nb)
        # Get random overlap matrix
        S  = self.get_random_overlap(nb)
        # Orthogonalise reference orbitals
        Cx = orthogonalise(Cx,S)

        nz = 1
        ran_h = np.random.randint(0,high=ne,size=nz)
        ran_p = np.random.randint(ne,high=nb,size=nz)
        ran_rot = np.random.rand(nb,nb)
        ran_rot -= ran_rot.T
        ran_rot[:,ran_h] = 0
        ran_rot[ran_h,:] = 0
        Cw = Cx.copy()
        Cw[:,[ran_h[0],ran_p[0]]] = Cx[:,[ran_p[0],ran_h[0]]]
        Cw = Cw.dot(expm(ran_rot))

        test_pair = gnme_pair(Cx,Cw,nc,ne,nb-nc,S)

        bstr = [x for x in cistring.make_strings(range(nb-nc),ne-nc)]
        for i in range(len(bstr)):
            for j in range(i,len(bstr)):
                bx, bw = bstr[i], bstr[j]

                # Get overlap from GNME
                wick_S = np.prod(test_pair.get_overlap(bx,bw))

                xhp, whp = test_pair.get_excitation_info(bx,bw)
                Cw_test = Cw.copy()
                for irow in range(whp.shape[0]):
                    i,a = whp[irow,0]+nc, whp[irow,1]+nc
                    Cw_test[:,[i,a]] = Cw[:,[a,i]]
                Cx_test = Cx.copy()
                for irow in range(xhp.shape[0]):
                    i,a = xhp[irow,0]+nc, xhp[irow,1]+nc
                    Cx_test[:,[i,a]] = Cx[:,[a,i]]
                slater_S = np.linalg.det(Cx_test[:,:ne].T.dot(S).dot(Cw_test[:,:ne]))
                print("{:20.10f} {:20.10f}".format(wick_S,slater_S))

if __name__=='__main__':
    np.random.seed(7)
    np.set_printoptions(linewidth=10000,suppress=True,precision=6)
    unittest.main()
