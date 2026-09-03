#!/usr/bin/python3
import numpy as np
from .pesObj import PESWalker
from .pesUtils import *
from .pesUtils import _pid_alive   # star-import skips leading-underscore names
import multiprocessing as mp
from multiprocessing.managers import SyncManager
import queue as _queue
import subprocess
import glob
import os, sys, uuid, signal,  threading, time

class SolutionRegistry:
 
    def __init__(self, value_of, merges_only=False):
        # Limited to 99999 solutions < really this should be 9999 instead only changed for current job >  
        # glob from all solutions in current directories
        # replaces append to used names and is general 
        self.freeVals = set(range(1,100000))
        self.seeds = self.read_all_solutions()  

        self.MERGES_FILE = "merges.txt"
        self.merge_claims = [] 
        if merges_only and os.path.exists(self.MERGES_FILE):
            with open(self.MERGES_FILE, "r") as f: 
                for line in f: 
                    data = line.split()    
                    self.merge_claims.append((data[0], data[1])) 
        else:
            open(self.MERGES_FILE, "w").close()

        self.suffixes = (".solution", ".hdf5", "_civec.txt")

        self.GEOMS_FILE = "geoms.txt"
        os.system(f"ls -d geom_* > {self.GEOMS_FILE}")
        geoms = np.genfromtxt("geoms.txt", dtype=str)
        ordering =np.argsort([value_of(x) for x in geoms ]) 
        self.geoms = list(geoms[ordering])

    def read_all_solutions(self):  
        summary = self.extract_info() 
        tmp_sols = [] 
        
        seeds = [] 
        if len(summary.shape)==1: 
            self.freeVals.remove(int(summary[0]))
            seeds.append((summary[0], summary[3], summary[5]))  
        else: 
            for isol, sol in enumerate(summary[:,0]):
                if str(sol)[0]=="t": 
                    tmp_sols.append((sol,summary[isol, 3])) 
                else:
                    self.freeVals.remove(int(sol))
                    seeds.append((summary[isol,0], summary[isol,3], summary[isol,5]))  
        
        for prov, geom in tmp_sols: 
            name = self.clean_up(prov, geom)
            seeds.append((name, geom, geom)) 
        # This returns a list of (sol, start_geom, end_geom) - so that we can set off the correct direction!
        return seeds 

    def extract_info(self):
        extract_solutions(out="extracted_solutions.txt")  
        summary = np.genfromtxt("extracted_solutions.txt", dtype=str) 
        return summary 
     
    def merge_geom(self, name, value_of):           
        if name in self.geoms:
            return False
        values = [value_of(g) for g in self.geoms]
        self.geoms.insert(int(np.searchsorted(values, value_of(name))), name)
        tmp = self.GEOMS_FILE + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(self.geoms) + "\n")
        os.replace(tmp, self.GEOMS_FILE)   
        return True


    def next_name(self):
        if len(self.freeVals)==0:
            n = None  
        else: 
            n = f"{self.freeVals.pop():04d}"
            ## this chooses the name 
        return n


    def record_claim(self, sol, onto, geom):
        """ Add pair of solutions that are actually the same and must be merged to merged_claims"""
        self.merge_claims.append((sol, onto))
        with open(self.MERGES_FILE, "a") as fh:
            fh.write(f"{sol} {onto} {geom}\n")
        return 


    def file_replace(self, geom, sourceSol, tarSol): 
        with geom_lock(geom):
            for suf in self.suffixes:
                source, target = f"{geom}/{sourceSol}{suf}", f"{geom}/{tarSol}{suf}"
                if not os.path.exists(source):
                    continue
                if os.path.exists(target):
                    os.unlink(source)
                else:
                    os.replace(source, target)
        return   
    

    def clean_up(self, prov, geom):
        name = self.next_name()
        if name:
            self.file_replace(geom, prov, name) 
        else: 
            #put into another queue 
            pass 
        return name


    def consolidate_merges(self):
        """ Collapses merged claims onto the solution with the smallest number"""
        if len(self.merge_claims)==0: 
            print(" No merge claims to process", flush=True) 
            return {}
 
        parent_of = {}
        def find(x):
            # This only adds into dictionary if x is not already a key... 
            parent_of.setdefault(x, x)
            root = x
            while parent_of[root] != root:
                root = parent_of[root]
    
            while parent_of[x] != root:          
                parent_of[x], x = root, parent_of[x]
            return root
    
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            lo, hi = (ra, rb) if int(ra) <= int(rb) else (rb, ra)
            parent_of[hi] = lo
    
        for sol, onto in self.merge_claims:
            union(sol, onto)
        
        # this builds the mapping from all the solution names to their root 
        canonical = {name: find(name) for name in parent_of}
        #for sol in sorted(canonical):
        #    target = canonical[sol]
        #    if target == sol:
        #        continue
        #    for path in sorted(glob.glob(f"*/{sol}.solution")):
        #        geom = os.path.dirname(path)
        #        self.file_replace(geom, sol, target) 
        #        moved += 1
        #    print(f"consolidated {sol} -> {target} ({moved} geometries)",
        #          flush=True)
        
        counts={} 
        for geom in glob.glob("geom_*"): 
            for path in glob.glob(f"{geom}/*.solution"):
                sol = os.path.basename(path)[:-len(".solution")]                
                target = canonical.get(sol)
                if target:
                    if target!=sol: 
                        if not counts.get(sol): 
                            counts[sol] = 1
                        else: 
                            counts[sol] += 1 
                        self.file_replace(geom, sol, target) 

        for sol in counts: 
            print(f"  {sol} -> {canonical[sol]} at {counts[sol]} geometries", flush=True)  
        return canonical


    def write_solution_summary(self, survivors, canonical, path="solutions.txt"):
        """One line per surviving solution with the span it covers, plus the
           retired -> canonical map."""
        lines = []
        for name in survivors:
            # >>>>>>>>>>>>>>>>>>> this is going to need to change! 
            vals = sorted(PESWalker.geom_value(os.path.dirname(p))
                          for p in glob.glob(f"*/{name}.solution"))
            span = f"{vals[0]:8.1f} .. {vals[-1]:8.1f}" if vals else "(no points)"
            lines.append(f"{name}  {len(vals):4d} points  {span}")
        retired = sorted(n for n, c in canonical.items() if c != n)
        if retired:
            lines.append("")
            lines.append("retired -> canonical")
            lines.extend(f"  {n} -> {canonical[n]}" for n in retired)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        os.replace(tmp, path)
        print(f"wrote {path}: {len(survivors)} live solutions, "
              f"{len(retired)} retired", flush=True)
        return

#-------------------------------------------------------

class TaskPool: 

    def __init__(self, proQueue, pool, pes_config, stop, merges_only=False):
        self.proQueue = proQueue
        self.pool = pool
        self.pes_config = pes_config 
        self.outstanding = {} 
        self.second_queue = [] 
        self.solution_registry = SolutionRegistry(PESWalker.geom_value, merges_only)
        self.stop = stop
        self.QUEUE_POLL = 20.0 

    def dispatch_walker(self, sol, geom, parent=None, fwd_only=None):
        dirs = None
        if fwd_only is None: 
            dirs = [ True, False]
            initSearches = dirs  
        else: 
            dirs = [ fwd_only ]
            initSearches = [ True ]  
 
        for ifwd, fwd in enumerate(dirs):
            tid = uuid.uuid4().hex[:8]
            self.outstanding[tid] = {
                "pid": None,
                "what": f"{sol} @ {geom} {'fwd' if fwd else 'bwd'}",
                "sol": sol, 
                "geom": geom, 
                "type": f"{'fwd' if fwd else 'bwd'}",
            }
            self.pool.apply_async(
                PESWalker(sol, self.solution_registry.geoms, self.proQueue, self.pes_config, parent, tid).construct_PES,
                (geom, fwd, initSearches[ifwd]),
                callback=lambda ret, tid=tid: self.proQueue.put(
                    ("done", {"tid": tid, "geoms": ret})),
                error_callback=lambda exc, tid=tid: self.proQueue.put(
                    ("failed", {"tid": tid, "exc": repr(exc)})),
            )
        return 


    def dispatch_sweep(self, solName):
        tid = uuid.uuid4().hex[:8]
        self.outstanding[tid] = {"pid": None, "what": f"sweep {solName}", "type": "sweep", "sol": solName}
        self.pool.apply_async(
            PESWalker(solName, self.solution_registry.geoms, self.proQueue, self.pes_config, None, tid).fill_solution_gaps,
            (),
            callback=lambda ret, tid=tid: self.proQueue.put(
                ("done", {"tid": tid, "geoms": []})),
            error_callback=lambda exc, tid=tid: self.proQueue.put(
                ("failed", {"tid": tid, "exc": repr(exc)})),
        )
        return

 
    def pump(self):
        """Single message loop, used for both the propagation phase and
           the sweep.  Polls rather than blocking forever so that a lost
           task (P1) and a shutdown request (P4) are both noticed."""
        while self.outstanding and not self.stop.is_set():
            try:
                kind, payload = self.proQueue.get(timeout=self.QUEUE_POLL)
            except _queue.Empty:
                self.reap_lost_tasks()
                continue

            if kind == "alive":
                tid = payload["tid"]
                if tid in self.outstanding:
                    self.outstanding[tid]["pid"] = payload["pid"]
                    self.outstanding[tid]["log"] = payload.get("log")
                continue

            if kind == "merged":
                self.solution_registry.record_claim(payload["sol"], payload["onto"],
                             payload["geom"])
                print(f"merge claim: {payload['sol']} == "
                      f"{payload['onto']} at {payload['geom']}")
                continue

            if kind == "found":
                print("message: ", kind, payload)
                prov, geom = payload["prov"], payload["geom"]
                parent = payload["parent"]
                if self.solution_registry.merge_geom(geom, PESWalker.geom_value):
                    print(f"grid refined: {geom} inserted")
                new_name = self.solution_registry.clean_up(prov, geom)
                if new_name is None:
                    self.second_queue.append((prov, geom, parent))
                    print(f"Setting aside branch {prov} at {geom}", flush=True)
                else:
                    self.dispatch_walker(new_name, geom, parent)
                continue

            tid = payload["tid"]
            try: 
                info = self.outstanding.pop(tid)
                label = info["what"] if info else tid
                if kind == "done":
                    print(f"message: done [{label}]")
                    for newgeom in payload["geoms"]:
                        if self.solution_registry.merge_geom(newgeom, PESWalker.geom_value):
                            print(f"grid refined: {newgeom} inserted")
                    os.replace(f"logs/{tid}_{info['type']}.log", f"logs/{info['sol']}_{info['type']}.log")
                    
                elif kind == "failed":
                    log = (info or {}).get("log")
                    print(f"task failed [{label}]: {payload['exc']}"
                          + (f" - see {log}" if log else ""))
            except: 
                print(f"task has no info?")
                
        return


    def consolidate_solutions(self):
        # This read it from solution_registry.merge_claims()  
        canonical = self.solution_registry.consolidate_merges()
        return canonical 


    def consolidate_and_sweep_solutions(self):
        canonical = self.consolidate_solutions() 
        survivors = self.solution_registry.extract_info()
        if len(survivors.shape)==1: 
            survivors = [ survivors[0] ]
        else: 
            survivors = survivors[:,0]
  
        self.solution_registry.write_solution_summary(survivors, canonical)
        for solName in survivors:
            print("submitting ", solName)
            self.dispatch_sweep(solName)
        self.pump()
        if self.outstanding:
            print("unfinished tasks after consolidate and sweep: ",
                  [i["what"] for i in self.outstanding.values()], flush=True)
        return 


    def reap_lost_tasks(self):
        for tid, info in list(self.outstanding.items()):
            pid = info["pid"]
            if pid is not None and not _pid_alive(pid):
                print(f"task {tid} [{info['what']}] died with worker pid {pid} "
                      f"- reaping", flush=True)
                self.outstanding.pop(tid, None)
        return


    def _kill_pool_workers(self,timeout=5.0):
        workers = list(getattr(self.pool, "_pool", []))
        for p in workers:
            if p.exitcode is None:
                try:
                    p.kill()
                except Exception:
                    pass
        deadline = time.time() + timeout
        for p in workers:
            try:
                p.join(timeout=max(0.0, deadline - time.time()))
            except Exception:
                pass
        return [p.pid for p in workers if p.exitcode is None]

    
    def shutdown_pool(self): 
        survivors = self._kill_pool_workers()
        if survivors:
            print("workers that would not die: ", survivors, flush=True)
        return
