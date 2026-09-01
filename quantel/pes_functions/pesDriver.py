#!/usr/bin/python3
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SyncManager
from .helperClasses import SolutionRegistry, TaskPool
import os, sys, threading, signal, traceback
from .pesUtils import _worker_init, extract_solutions

def pesDriver(config, seeds): 
    # Create directory to hold the .lock files 
    os.makedirs("temp", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.system("rm geom_*/tmp*") 
    stop = threading.Event()
    def _terminate(signum, _frame):
        stop.set()
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT,  _terminate)
    #--------------------------------
    
    def make_geom_name(value):
        if config["geoms"]["includeSign"]:  
            return f"geom_{value:+0{ config['geoms']['geomDecimals'] + config['geoms']['leading_zeros'] + 2 }.{config['geoms']['geomDecimals']}f}"
        else: 
            return f"geom_{value:0{ config['geoms']['geomDecimals'] + config['geoms']['leading_zeros'] + 1 }.{config['geoms']['geomDecimals']}f}"
    
    #---- Setup -----
    manager = SyncManager()
    manager.start(initializer=_worker_init)
    proQueue = manager.Queue()
    pool = mp.Pool(processes=config["jobcontrol"]["NPROC"], maxtasksperchild=1, initializer=_worker_init)
    try:
        try: 
            if config["jobcontrol"]["propagate_solutions"]:  
                taskPool = TaskPool(proQueue, pool, config, stop)
                if len(seeds)==0: 
                    extract_solutions(out = "extracted_solutions.txt")
                    solInfo = np.genfromtxt("extracted_solutions.txt", dtype=str) 
                    
                    for i in range(solInfo.shape[0]):
                        if solInfo[i,3] == solInfo[i,5]:
                            seeds.append((solInfo[i,0],make_geom_name(float(solInfo[i,3]))))
                        else:                                                               
                            seeds.append((solInfo[i,0],make_geom_name(float(solInfo[i,3])))) 
                            seeds.append((solInfo[i,0],make_geom_name(float(solInfo[i,5])))) 
                                        
                for nsol, ngeom in seeds: 
                    taskPool.dispatch_walker(nsol,ngeom)
                taskPool.pump() 
                if stop.is_set():
                        print("shutdown requested - skipping solution sweep", flush=True)
                else:
                    taskPool.consolidate_and_sweep_solutions()
            elif config["jobcontrol"]["consolidate_only"]: 
                taskPool = TaskPool(proQueue, pool, config, stop, merges_only=True) 
                taskPool.consolidate_solutions() 
            else: 
                taskPool = TaskPool(proQueue, pool, config, stop, merges_only=True) 
                taskPool.consolidate_and_sweep_solutions()
        finally:
            taskPool.shutdown_pool()
        print(" all propagations finished", flush=True)
        status = 0
    except BaseException:
        traceback.print_exc()
        status = 1
    finally:
        manager.shutdown()
        # Hard exit on every path.  A normal interpreter shutdown would try to join
        # the pool workers, and the pool replaces any worker shutdown_pool() kills
        # (maxtasksperchild=1), so the join can block forever.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)
