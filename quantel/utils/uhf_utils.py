from quantel.wfn.uhf import UHF 
import numpy as np 
import glob, copy  

def include_spin_flips(wfnlist, jobcontrol_config ):
    # Assumes solutions are numbered
    if jobcontrol_config["save_solns"]:
        namelist = glob.glob("*.solution")
        namelist = [ x[:-9] for x in namelist ] 
        count = max([ int(x) for x in namelist])

    for i, wfn in enumerate(wfnlist):
        wfn.update()
        flip = wfn.get_spin_flip()

        new = True 
        for previous_soln in wfnlist: 
            if (np.abs(previous_soln.energy-flip.energy)<jobcontrol_config["dist_thresh"]): 
                if (1-np.abs(previous_soln.overlap(flip))<jobcontrol_config["dist_thresh"]): 
                    new = False
                    break  
        if new: 
            print(f"Unique spin flip located for Solution {i} in list")
            if jobcontrol_config["save_solns"]: 
                count += 1 
                flip.save_to_disk(f"{count:04d}")
            wfnlist.append(flip) 
    return
