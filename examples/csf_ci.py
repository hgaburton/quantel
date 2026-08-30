from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals 
from quantel.wfn.csf import CSF 
import numpy as np 
import itertools 
from quantel.utils.csf_utils import csf_to_cimat  


mol = PySCFMolecule("/home/cyzbrown/code/quantel/examples/mol/formaldehyde.xyz", "aug-cc-pvdz", "angstrom", spin=0,charge=0) 
ints = PySCFIntegrals(mol) 


wfn = CSF(ints, "+-")

#def csf_to_cimat(spin_coupling,ncore,nmo):
#    """ Get the list of open-shell determinants for a given CSF pattern
#            :param spin_coupling:
#            :return:
#    """
#    if (spin_coupling!=""): 
#        if(spin_coupling[0]!='+') :
#            raise RuntimeError("Invalid spin coupling pattern")
#    
#    nactive = len(spin_coupling)  
#    nvir = nmo - ncore - nactive
#
#    from itertools import combinations
#    def strings_with_n_ones(length, n_ones):
#        results = []
#        for positions in combinations(range(length), n_ones):
#            s = [0] * length
#            for p in positions:
#                s[p] = 1
#            results.append(s)
#        return sorted(results)
#    
#    csf_Tn = [] 
#    prev = 0  
#    for ispin,spin in enumerate(spin_coupling): 
#        csf_Tn.append( 0.5+prev if spin=="+" else -0.5+prev ) 
#        prev = csf_Tn[ispin] 
#
#    if len(csf_Tn)!= 0 : 
#        nalfa = int(csf_Tn[-1]+nactive/2) 
#    else: 
#        nalfa = 0 
#    nbeta = nactive - nalfa 
#    alfa_basis = strings_with_n_ones(nmo, nalfa+ncore)
#    beta_basis = strings_with_n_ones(nmo, nbeta+ncore)
#    
#    #for s in alfa_basis:
#    #    print(s) 
#    #print("---------------------")
#    #for s in beta_basis:
#    #    print(s) 
#    
#    ci_mat = np.zeros((len(alfa_basis), len(beta_basis)), dtype=float) 
#    for i in range(len(alfa_basis)): 
#        for j in range(len(beta_basis)):
#            if not ( (alfa_basis[i][:nvir] == [ 0 for _ in range(nvir) ]) and (beta_basis[j][:nvir] == [ 0 for _ in range(nvir) ])):
#                continue 
#            if not ( (alfa_basis[i][nvir+len(spin_coupling):] == [ 1 for _ in range(ncore) ]) and (beta_basis[j][nvir+len(spin_coupling):] == [ 1 for _ in range(ncore) ])):
#                continue 
#            
#            Pn = [] 
#            prev = 0 
#            for k in range(nvir+len(spin_coupling)-1,nvir-1,-1):
#                coeff =  prev + 0.5*(alfa_basis[i][k]-beta_basis[j][k])
#                Pn.append(coeff )
#                prev = coeff
#           
#            ci_mat[i,j] = get_total_coupling_coefficient(Pn, csf_Tn)  
#
#    return ci_mat  


cimat,alfa_basis, beta_basis = csf_to_cimat("+-")
print(cimat) 
