'''
Example for computing Slater-Condon rules with factorised NOCI 2RDM
'''
from quantel.gnme.utils import factorised_densities
from quantel.utils.linalg import orthogonalise
from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
import numpy as np

if __name__ == "__main__":
    # This PySCFMolecule inherits from a PySCF gto.Mole object
    mol  = PySCFMolecule([['Li',0,0,0],['Li',0,0,6]],'sto-3g','bohr')
    # This integral object is basically responsible for handling interface to AO integrals 
    ints = PySCFIntegrals(mol)
    # Get the AO integrals
    metric = ints.overlap_matrix()
    hcore  = ints.oei_matrix()

    # Make some fake orbital coefficients
    Cwa = orthogonalise(np.random.rand(ints.nbsf(),ints.nmo()), metric)[:,:3] 
    Cwb = orthogonalise(np.random.rand(ints.nbsf(),ints.nmo()), metric)[:,:3]
    Cxa = orthogonalise(np.random.rand(ints.nbsf(),ints.nmo()), metric)[:,:3]
    Cxb = orthogonalise(np.random.rand(ints.nbsf(),ints.nmo()), metric)[:,:3]


    ### Compute the factorised densities and overlap
    # S = overlap
    # (Wa, Ma, Pa) are alpha densities
    # (Wb, Mb, Oa) are beta densities
    # kt is a vector of diagonal 
    S, (Wa, Ma, Pa), (Wb, Mb, Pb), kt = factorised_densities(Cwa,Cwb,Cxa,Cxb,metric)
    # Number of non-zero pure two-body terms 
    nd = 2 + len(kt)

    ### Compute energy coupling with PySCF
    H = S * ints.scalar_potential()

    # One-body contribution
    H += S * np.einsum('ij,ji', hcore, Wa + Wb) + np.einsum('ij,ji', hcore, Ma + Mb)

    # Setup variables for JK builds
    vd = np.zeros((2*nd,ints.nbsf(),ints.nbsf()))
    vd[0] = Wa
    vd[1] = Ma
    vd[2:nd] = Pa
    vd[nd] = Wb
    vd[nd+1] = Mb
    vd[nd+2:] = Pb

    # Call a JK build for each density
    vj, vk = ints.build_multiple_JK(vd,vd)

    # Compute two-electron contributions
    jw_t = vj[0] + vj[nd]
    Wt = Wa + Wb
    Mt = Ma + Mb
    H += 0.5 * np.einsum('ij,ji',jw_t, S * Wt + 2 * Mt)
    H -= 0.5 * np.einsum('ij,ji',vk[0],  S * Wa + 2 * Ma)
    H -= 0.5 * np.einsum('ij,ji',vk[nd], S * Wb + 2 * Mb)
    for t, ek in enumerate(kt):
        Pk = Pa[t] + Pb[t]
        Jk = vj[2+t] + vj[nd+2+t]

        H += 0.5 * ek * np.einsum('ij,ji',Jk, Pk)
        H -= 0.5 * ek * np.einsum('ij,ji',vk[2+t], Pa[t])
        H -= 0.5 * ek * np.einsum('ij,ji',vk[nd+2+t], Pb[t])
