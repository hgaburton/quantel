I have used four python scripts to produce results for this project.

1) "NR_CASSCF.py"

This script is the principal one that run the grid search.
To use it you need the following command:
python3 NR_CASSCF.py molecule.xyz input.txt
I have put some examples of input files in the git repository.

I was writing the output of this script in a result.txt file (see results folder in the git repo for an example)

2) "analyze_result.py"

This script is used to read the previous result.txt file and the output is a list of the different solutions found during the grid search.
To use it you need the following command:
python3 analyze_result.py result.txt

If you use
python3 analyze_result.py result.txt molecule.xyz input.txt
It will also check the scalar product between each solution close in energy and remove the duplicates (scalarprod=1)

3) "PES.py"

This one does not take argument for the moment... You have to modify the file and then run it using
python3 PES.py
You have to give a result file as an input as well as the number of the calculation in this file that you want to use as a starting point for the PES.

Below this code for SS-CASSCF PES, there are also two parts commented which were used to draw the FCI and SA-CASSCF PES.

Finally, the last commented part correspond to the script used to follow a negative eigenvalue and find the "adiabatic" LiF solution. This script is not general and can only be used for this solution for the moment ...

4) "plot_orbitals.py"

This script is used to plot the orbitals in the cube file format.


#######################################################################################################

Configuration State Function (CSF) optimisation

#######################################################################################################

Example files can be found in input/h4

Example command:
python search_csf.py h4_1.0.xyz h4_config.txt

There are some keywords relevant to CSF optimisation
active: Indices of the active orbitals (0-th indexing). In the example, 0 1 2 3 refers to using the first 4 orbitals.

g_coupling: The genealogical coupling of the CSF desired. Every S=0.5 is denoted by + and every S=-0.5 is denoted by -. e.g. M CSF is given by +-+- and V CSF is given by ++--

permutation: The orbital arrangement of the active orbitals (0-th indexing).

mo_basis: The initial MO used when initialising the CSF. Current options are:
	''site'': Symmetrically orthogonalised orbitals (site basis)
	''hf'': (Restricted) Hartree--Fock orbitals
	''custom'': Uses a custom set of orbitals. This reads the orbitals from a custom_mo.npy file in the worki		     ng directory

Running the example command given, one should eventually arrive at 3 unique solutions:
-1.168224
-1.197117
-1.181668

These correspond to 1-2/3-4 (Case A), 1-3/2-4 (Case B), and 1-4/2-3 (Case C) spin coupling patterns respectively.
