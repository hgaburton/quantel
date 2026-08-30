import numpy as np
import os, sys, glob, ctypes
import signal, threading, time
import copy

# Extract Utils
def extract_solutions(out = "extracted_solutions.txt"): 
    command = """ 
    find . -mindepth 2 -maxdepth 2 -name '*.solution' -printf '%f\t%h\n' |
    sed 's|\.solution\t\./geom_|\t|' |
    sort -t"$(printf '\t')" -k1,1 -k2,2n |
    awk -F'\t' '
        $1 != prev {
            if (n) print prev "  " n " points  " first " .. " last
            prev = $1; n = 0; first = $2
        }
        { n++; last = $2 }
        END { if (n) print prev "  " n " points  " first " .. " last }
    '  >  """ + " extracted_solutions.txt" 
    try: 
        os.system(command) 
        return True 
    except: 
        return False  

# Paralleliser Utils 
# --- teardown plumbing: make children die with the parent --
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_PR_SET_PDEATHSIG = 1
_PPID_POLL = 5.0
def _worker_init():
    _libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    # Drop the parent's SIGTERM handler, inherited across the fork.  Only the
    # driver has a loop that reads the stop flag; here it just makes the worker
    # deaf to the terminate the interpreter sends at shutdown, so the parent
    # blocks forever joining it.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    ppid = os.getppid()
    if ppid == 1:
        os._exit(1)

    def _watch():
        while os.getppid() == ppid:
            time.sleep(_PPID_POLL)
        os._exit(1)
    threading.Thread(target=_watch, daemon=True, name="ppid-watchdog").start()

def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

#------------------------------------------------
# Naming/Saving functions
#------------------------------------------------
def read_solution_header(path):
    """Return (energy, hess_index) from a .solution file."""
    with open(path, "r") as f:
        line = f.readline().split()
    return float(line[0]), int(line[1])


def saved_solutions_at(geom):
    """List of (name, energy, index) for every solution saved at geom."""
    out = []
    for path in glob.glob(geom + "/*.solution"):
        name = os.path.basename(path)[:-len(".solution")]
        e, ind = read_solution_header(path)
        out.append((name, e, ind))
    return out


def match_known_solution(geom, wfn, dedupEnergyThresh, dedupOverlapThresh,
                         exclude=None):
    """Name of an already-saved solution at geom that is the same solution as
       `wfn`, or None.

       Identity requires all three of: matching Hessian index, energy within
       dedupEnergyThresh, and 1-|overlap| within dedupOverlapThresh. The two
       tolerances are deliberately different"""
    for name, e, ind in saved_solutions_at(geom):
        if name == exclude:
            continue
        if ind != wfn.hess_index[0] or abs(e - wfn.energy) >= dedupEnergyThresh:
            continue
        other = wfn.copy()
        other.read_from_disk(geom + "/" + name)
        ovlp = np.abs(wfn.overlap(other))
        # 1-|ovlp| in full precision: at :.6f on |ovlp| every one of these
        # decisions printed as 1.000000, which hid the deciding digits.
        print(f"  identity check vs {name}: dE={abs(e - wfn.energy):.2e} "
              f"1-|ovlp|={1.0 - ovlp:.3e}")
        if 1.0 - ovlp < dedupOverlapThresh:
            return name
        print(f"  -> degenerate with {name} but distinct (low overlap)")
    return None


#------------------------------------------------
# Hessian analysis
#------------------------------------------------
def analyse_hessian(wfn, hessThresh):
    """Return (eigval, eigvec, currHessInd, zeroInds) for wfn's Hessian
       and set the currHessIndex as wfn.hess_index"""
    eigval, eigvec = np.linalg.eigh(wfn.hessian)
    currHessInd = int(np.sum(eigval < -hessThresh))
    wfn.hess_index = (currHessInd, 0)
    return eigval, eigvec, currHessInd


def pick_search_index(eigval, currHessInd, hessZeroThresh):
    """Target index for the coalescing-partner search implied by the softest
       zero mode, and the zero mode's position. (None, None) if there is no
       coalescence signature."""

    zeroInds = np.argwhere(np.abs(eigval) < hessZeroThresh).reshape((-1))
    if len(zeroInds) == 0:
        #return None, None 
        return [] 

    # Choose the eigvector direction with smallest magnitude
    #zi = zeroInds[np.argsort(np.abs(eigval[zeroInds]))[0]] 
    # order by smallest magnitude but keep both 
    Zis = zeroInds[np.argsort(np.abs(eigval[zeroInds]))] 
    searchInfo = [] 
    for zi in Zis: 
        if (zi == currHessInd - 1):
            searchInfo.append((zi, zi))
        elif (zi == currHessInd ):
            searchInfo.append((zi+1, zi))
        else:
            pass 
    return searchInfo 
    #if (zi == currHessInd - 1):
    #    return zi, zi
    #elif (zi == currHessInd ):
    #    return zi+1, zi
    #else:
    #    return None, None


def report_hessian(geom, prevHessInd, eigval, currHessInd):
    #print(f"Geometry {geom}")
    #print(f"Prev hess ind {prevHessInd}")
    #print("Hessian eigenvalues: ", eigval[:10])
    #print(f"Current Hessian index {currHessInd}")
    return


#---------------------------------
# Multiprocessing functions
#---------------------------------
import os, time, glob, fcntl, contextlib, uuid, random, shutil
@contextlib.contextmanager
def geom_lock(path):
    #os.makedirs(path, exist_ok=True)
    lockfile = "temp/.lock"+path
    print("LOCKING : ", lockfile, flush=True)
    with open(lockfile, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
