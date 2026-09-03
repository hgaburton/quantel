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
    
    def __init__(self, wfn, ncas, nact_alfa, nact_beta):
        # this works for the mixing of cs and +- - test that first? 
        self.mol = wfn.integrals.mol()
        self.fcisolver = fcisolver() 
        self.nelec=(wfn.nalfa+wfn.nbeta) 
        self.ncore = int((self.nelec - nact_alfa - nact_beta)/2) 
        self.ncas = ncas 
        self.mo_coeff=(wfn.mo_coeff.copy())
        self.ci, _ , _ = csf_to_cimat(wfn.spin_coupling,ncas, nact_alfa, nact_beta) 
        print(f" spin coupling: ({wfn.spin_coupling}), self.ci: ", self.ci)  
        self.energy=(wfn.energy)  
       
    def kernel(self): 
        return [self.energy]

def evcont_interpolation(config, sampleGeoms, finalGeoms, solNames):    
    dummy_states = [ ] 
    sampleMols = []
    finalMols = []
    learningSpace = [] 
    nroots=config["nroots"]
    ncas =  config["ncas"]
    nact_alfa = config["nact_alfa"] 
    nact_beta = config["nact_beta"]  
    neleca = nact_alfa + nact_beta
    os.makedirs("temp_geoms/", exist_ok = True)  
    for geom in finalGeoms:
        path = f"{geom}/geom.xyz"  
        if not os.path.exists(path):
            # Surely a benefit is that we dont need to find solutions at all geometries 
            geomVal = float(geom[5:]) 
            os.system(f"python {config['makeGeomScript']} {geomVal} > temp_geoms/{geom}.xyz") 
            path = f"temp_geoms/{geom}.xyz" 
             
        mol = PySCFMolecule(path, config["basis"], config["units"]) 
        ints = PySCFIntegrals(mol) 
        finalMols.append(mol)  
    for geom in sampleGeoms:  
        print(f"  Construct mol for {geom}/geom.xyz", flush=True)
        if not os.path.exists(f"{geom}/geom.xyz"): 
            continue 
        mol = PySCFMolecule(f"{geom}/geom.xyz", config["basis"], config["units"])
        print("Natom :", mol.natom())  
        ints = PySCFIntegrals(mol) 
        wfn = CSF(ints,"+-")
        states = []
        for sol in solNames: 
            if os.path.exists(f"{geom}/{sol}.solution"): 
                print(f"  Sampled {geom}/{sol}", flush=True)
                learningSpace.append(f"{geom}/{sol}")
                wfn.read_from_disk(f"{geom}/{sol}")
                states.append(DummyState(wfn, ncas, nact_alfa, nact_beta)) 
        dummy_states.append(states) 
        sampleMols.append(mol)
   
    natom = sampleMols[0].natom()     
 
    lowrank_kwargs = {
        "truncation_style": "eigval",
        "eval_thr": 1e-12,
        "save_diag": False,
    }
    cont_lr = CAS_EVCont_obj(
        ncas,
        neleca,
        nroots ,
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
   
    vectors = np.zeros((len(finalMols), nroots, len(learningSpace)), dtype = float)  
    for ifmol, fmol in enumerate(finalMols):
        print("================================") 
        print(f" Approximating for {finalGeoms[ifmol]}")  
        energies[ifmol,1:], vecs = approximate_multistate_lowrank_OAO(
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
        print(" Vectors: ")
        print(vecs)
        vectors[ifmol, :, : ] = vecs  
    print("================================") 
    print("Learning space labels") 
    print(learningSpace) 
 
    np.save("evcont_vecs.npy", vectors) 
    np.savetxt("evcont_learningSpace.txt", learningSpace, fmt="%s")
    np.savetxt("evcont_energies.txt", energies) 
    return 
