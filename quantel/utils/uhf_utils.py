from quantel.wfn.uhf import UHF 
import numpy as np 
import glob, copy  

def include_spin_flips(wfnlist, nlist):
    fnlist = [] 
    ilist = [] 
    elist = [] 
    flip_wfnlist = [] 
    
    # Creating names - 
    # Assumes solutions are numbered as --0018
    count = max([ int(x[-4:]) for x in nlist])
    intlist = [ int(x[-4:]) for x in nlist]
    print("count: ", count)    
    print("intlist: ", intlist)    
 
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
            temp = f"{count:04d}"
            flip_wfnlist.append(flip)
            found = True 
            #for i in range(10000): 
            #    temp = f"{i:04d}"
            #    if not (temp in nlist): 
            #        found = True 
            #        break
 
            if found: 
                fnlist.append(temp)
                elist.append(flip.energy) 
                ilist.append(flip.hess_index[0])
                print(f"  Spin flip saved as {temp}", flush=True) 
            else: 
                print("  Couldn't find unused name - spin flip was not saved", flush=True) 
    return flip_wfnlist, fnlist, elist, ilist  
