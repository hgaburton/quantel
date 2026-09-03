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
                    seeds = taskPool.solution_registry.seeds 
                
                for nsol, init_geom, final_geom in seeds: 
                    print(f"{nsol}  {init_geom} - > {final_geom}", flush = True) 
                    taskPool.dispatch_walker(nsol, f"geom_{init_geom}",  fwd_only=False)
                    taskPool.dispatch_walker(nsol, f"geom_{final_geom}", fwd_only=True)
                
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
