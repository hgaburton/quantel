import quantel 
from quantel.utils.scf_utils import sorting_shells 
from quantel.wfn.csf import CSF
from quantel.wfn.rhf import RHF
from quantel.opt.diis import DIIS 
import numpy as np 
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals 
from quantel.opt.lbfgs import LBFGS

mol = PySCFMolecule('mol/formaldehyde.xyz','aug-cc-pvdz','angstrom')
ints = PySCFIntegrals(mol)

wfn= RHF(ints, 'cs')
wfn.get_orbital_guess(method="gwh")
LBFGS().run(wfn)
#wfn.canonicalize()
wfn.localise()
ab2_orbs = wfn.get_AB2_orbitals()
wfn.update_vir_orbitals(ab2_orbs)
ref_mo = wfn.mo_coeff.copy()
old_coeff = wfn.mo_coeff.copy()
ref_mo[:,[7,25]] = ref_mo[:,[25,7]]
##Energies 
print("old coeff RHF:", wfn.energy)
wfn.initialise(mo_guess=ref_mo)
print("Ref mo RHF:", wfn.energy)


wfn = CSF(ints, 'cs')
wfn.initialise(mo_guess = old_coeff) 
projections1 = wfn.shell_projection(ref_mo, wfn.mo_coeff)
projections2 = wfn.shell_projection(old_coeff, wfn.mo_coeff)

Cord1, order1 = sorting_shells(wfn, projections1) 
Cord2, order2 = sorting_shells(wfn, projections2) 
print(order1)
print(order2)

inc_vir_projections1 = np.zeros((wfn.nmo, wfn.nshell+2))
inc_vir_projections2 = np.zeros((wfn.nmo, wfn.nshell+2))
inc_vir_projections1[:,:wfn.nshell+1] = projections1 
inc_vir_projections2[:,:wfn.nshell+1] = projections2 
 
total_proj1 = 0 
total_proj2 = 0 
for W, shell in enumerate(order1):
    for index in shell: 
        total_proj1 += inc_vir_projections1[index, W]
for W, shell in enumerate(order2):
    for index in shell: 
        total_proj2 += inc_vir_projections2[index, W]
print("Ordered1:", total_proj1) 
print("Ordered2:", total_proj2) 

##Energies 
coeff = wfn.mo_coeff.copy()
od1 = [i for shell in order1 for i in shell ] 
od2 = [i for shell in order2 for i in shell ] 
wfn.initialise(mo_guess=coeff[:,od2])
print("old coeff CSF:", wfn.energy)
wfn.initialise(mo_guess=coeff[:,od1])
print("Ref mo CSF:", wfn.energy)

def mom_select(Cocc, Cnew, metric):
    p = np.einsum('pj,pq,ql->l', Cocc,metric,Cnew,optimize="optimal")
    # Order MOs according to largest projection 
    idx = list(reversed(np.argsort(np.abs(p))))
    idx = [ int(x) for x in idx ]
    return Cnew[:,idx], idx 

Cord1, morder1 = mom_select(ref_mo[:,:8], wfn.mo_coeff, wfn.integrals.overlap_matrix() ) 
Cord2, morder2 = mom_select(old_coeff[:,:8], wfn.mo_coeff, wfn.integrals.overlap_matrix() ) 
print(morder1)
print(morder2)

proj1=0
count=0 
for W,shell in enumerate(order1): 
    for i in range(len(shell)): 
        proj1 += inc_vir_projections1[ morder1[count], W]
        count += 1 

proj2=0
count=0 
for W, shell in enumerate(order1): 
    for i in range(len(shell)): 
        proj2 += inc_vir_projections2[ morder2[count], W]
        count += 1 
print(proj1)
print(proj2)


wfn.initialise(mo_guess=coeff[:,morder2])
print("old coeff CSF -2:", wfn.energy)
wfn.initialise(mo_guess=coeff[:,morder1])
print("Ref mo CSf -2:", wfn.energy)






#################
#wfn= CSF(ints, '+-')
#wfn.get_orbital_guess(method="gwh")
#Cold = wfn.mo_coeff.copy()
#
#LBFGS().run(wfn)
#wfn.canonicalize()
#Cnew = wfn.mo_coeff.copy()
#
#projections = wfn.shell_projection(Cold, Cnew)
##returns an array of projection = [orbital, shell]
#
#Cord, order = sorting_shells(wfn, projections) 
#
#inc_vir_projections = np.zeros((wfn.nmo, wfn.nshell+2))
#inc_vir_projections[:,:wfn.nshell+1] = projections 
# 
#total_proj = 0 
#for W, shell in enumerate(order):
#    for index in shell: 
#        total_proj += inc_vir_projections[index, W]
#print("Ordered:", total_proj) 
# 
#
#orb_indices = np.arange(0, wfn.nmo)
#shell_indices = [wfn.core_indices] + wfn.shell_indices + [[ i for i in range(wfn.ncore + wfn.nopen, wfn.nmo)  ]]
#shell_lens = [ len(shell) for shell in shell_indices ]
#for _ in range(10): 
#    np.random.shuffle(orb_indices)
#    count=0 
#    rand_proj=0
#    for W,shell in enumerate(shell_indices): 
#        for ind in shell_lens:  
#            rand_proj += inc_vir_projections[ orb_indices[count], W]
#            count += 1
#    print("Random:", rand_proj) 
################

