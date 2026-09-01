#!/usr/bin/env python3
import glob, os  
import numpy as np
from evcont.CASCI_EVCont import CAS_EVCont_obj
from evcont.ab_initio_eigenvector_continuation import (
    approximate_multistate_OAO,
    approximate_multistate_lowrank_OAO,
)
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals  
from quantel.wfn.csf import CSF
from quantel.utils.csf_utils import csf_to_cimat 

class fcisolver(): 
    def __init__(self): 
        self.converged = True 

class DummyState(): 
    
    def __init__(self, wfn):
        # this works for the mixing of cs and +- - test that first? 
        self.mol = wfn.integrals.mol()
        self.fcisolver = fcisolver() 
        self.nelec=(wfn.nalfa+wfn.nbeta) 
        self.ncore = 7 
        self.ncas = 2  
        self.mo_coeff=(wfn.mo_coeff.copy())
        if wfn.spin_coupling == "" or wfn.spin_coupling == "cs": 
            #self.ci = np.array([[0,0],[0,1]], dtype=float)  
            self.ci = np.array([[1,0],[0,0]], dtype=float)  
        else: 
            self.ci, _ , _ = csf_to_cimat(wfn.spin_coupling) 
        print(f" spin coupling: ({wfn.spin_coupling}), self.ci: ", self.ci)  
        self.energy=(wfn.energy)  
       
    def kernel(self): 
        return [self.energy]

def evcont_interpolation(basis, units, nroots, sampleGeoms, finalGeoms, solNames):    
    dummy_states = [ ] 
    sampleMols = []
    finalMols = []
    for geom in finalGeoms:  
        mol = PySCFMolecule(f"{geom}/geom.xyz", basis, units) 
        ints = PySCFIntegrals(mol) 
        finalMols.append(mol)  
    for geom in sampleGeoms:  
        print(f"  Construct mol for {geom}/geom.xyz", flush=True) 
        mol = PySCFMolecule(f"{geom}/geom.xyz", basis, units)
        print("Natom :", mol.natom())  
        ints = PySCFIntegrals(mol) 
        wfn = CSF(ints,"+-")
        states = []
        for sol in solNames: 
            if os.path.exists(f"{geom}/{sol}.solution"): 
                print(f"  Sampled {geom}/{sol}", flush=True)
                wfn.read_from_disk(f"{geom}/{sol}")
                states.append(DummyState(wfn)) 
        dummy_states.append(states) 
        sampleMols.append(mol)
   
    ncas = 2 
    neleca = 2
    nelec = sampleMols[0].nalfa() + sampleMols[0].nbeta() 
    natom = sampleMols[0].natom()     
    
    print("natom", natom) 
 
    lowrank_kwargs = {
        "truncation_style": "eigval",
        "eval_thr": 1e-12,
        "save_diag": False,
    }
    cont_lr = CAS_EVCont_obj(
        ncas,
        neleca,
        nroots=nroots,
        solver="CASCI",
        lowrank=True,
        **lowrank_kwargs,
    )
    
    # Build training set.
    for ind, mol in enumerate(sampleMols):
        cont_lr.append_to_rdms(mol,dummy_states[ind])
    
    # Vectorize low-rank representation for fast inference.
    cont_lr.vectorize_lowrank(hermitian=True)
    # Low-rank representation details
    nvecs = cont_lr.lowrank_vectorized['nvecs']
    
    energies = np.zeros((len(finalMols),nroots+1), dtype=float)
    energies[:,0] = [ float(geom[5:]) for geom in finalGeoms ]    
    for ifmol, fmol in enumerate(finalMols):
        energies[ifmol,1:], _ = approximate_multistate_lowrank_OAO(
            fmol,
            cont_lr.one_rdm,
            cont_lr.lowrank_vectorized,
            cont_lr.diagonal_vectorized,
            cont_lr.overlap,
            nroots=nroots,
            density_fit=False,
            Jdiag_only=True,
            sao_diag=False,
        )
    np.savetxt("evcont_energies.txt", energies) 
    return 
