#!/usr/bin/python3
import numpy as np

def write_cidump(civec, ncore, nmo, filename,thresh=1e-10):
    """ Write a CI vector dump file from CI vector

        Input:
            civec: list of tuples (det, ci) where det is the determinant string and ci is the CI coefficient
            ncore: number of core orbitals
            nmo  : number of molecular orbitals
            filename: name of the output file
    """ 
    # Open output file for writing
    outF = open(filename,'w')

    # Loop over CI vector
    for det in civec:
        # Initialise occupation vectors
        alfa_occ = np.zeros(nmo,dtype=int)
        beta_occ = np.zeros(nmo,dtype=int)

        # Get occupation of core orbitals
        alfa_occ[:ncore] = 1
        beta_occ[:ncore] = 1

        # Get occupation of open-shell orbitals
        for i, di in enumerate(det[0]):
            if(di=='a' or di=='2'):
                alfa_occ[i+ncore] = 1
            if(di=='b' or di=='2'):
                beta_occ[i+ncore] = 1

        # convert list of 0 and 1 to bitstring
        alfa_bs = '0b'+(''.join(map(str,alfa_occ)))[::-1]
        beta_bs = '0b'+(''.join(map(str,beta_occ)))[::-1]

        # Write to file
        if(abs(det[1]) > thresh):
            outF.write(f'{alfa_bs}  {beta_bs}  {det[1]:20.16f}\n')

    # Close output file
    outF.close()

# NOCI utils

def get_chergwin_coulson_weights(wfn_list, evecs): 
    # Construct overlap matrix 
    ncsfs = len(wfn_list) 
    overlap_matrix = np.zeros((ncsfs, ncsfs))
    for i in range(ncsfs): 
        for j in range(i): 
            overlap_matrix[i,j] = wfn_list[i].overlap(wfn_list[j])    
    overlap_matrix += overlap_matrix.T
    np.fill_diagonal(overlap_matrix,1)
    # Calculate the Chergwin-Coulson weights 
    W = np.linalg.multi_dot((overlap_matrix, evecs))
    W = np.multiply(evecs, W)
    #W = [ W[i]*evecs[i,0] for i in range(ncsfs) ]
    return np.array(W), overlap_matrix













