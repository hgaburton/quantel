from quantel.wfn.uhf import UHF 
import numpy as np 
import glob, copy  

def include_spin_flips(wfnlist, nlist):
    fnlist = [] 
    ilist = [] 
    elist = [] 
    flip_wfnlist = [] 
    # Assumes solutions are numbered
    count = max([ int(x) for x in nlist])
    for i, wfn in enumerate(wfnlist):
        wfn.update()
        flip = wfn.get_spin_flip()
        new = True 
        for previous_soln in wfnlist: 
            if (np.abs(previous_soln.energy-flip.energy))<1e-8: 
                if (1-np.abs(previous_soln.overlap(flip)))<1e-8: 
                    new = False
                    break  
        if new: 
            print(f"Unique spin flip located for Solution {i} in list")
            flip.get_davidson_hessian_index() 
            count += 1 
            flip_wfnlist.append(flip) 
            fnlist.append(f"{count:04d}")
            elist.append(flip.energy) 
            ilist.append(flip.hess_index[0]) 
    return flip_wfnlist, fnlist, elist, ilist  
