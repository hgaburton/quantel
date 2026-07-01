import quantel 
import numpy as np
from guga_utils import * 
from quantel.utils.linalg import matrix_print 

def ndrt(a,b,c): 
    d = min(a,c)
    return (a+1)*(c+1)*(b+1+d/2)-(1/6)*d*(d+1)*(d+2) 
# so im constructing far more elements than necessary in the drt? 
# and i guess because im not considering what? 
conf1 = [3,3,3,3,1,2,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
n_elec = conf1.count(1) + conf1.count(2) + 2*conf1.count(3)
nmo = len(conf1) 
totspin = 0.5*(conf1.count(1)-conf1.count(2))

a = int(0.5*n_elec - totspin)  
b = int(2*totspin) 
c = int(nmo-a-b) 
cobj = quantel.Configuration(conf1)
print("ndrt", ndrt(a,b,c)) 
drt  = get_drt(a,b,c) 
print("drt shape", drt.shape) 
#print( drt) 
print("==========") 
space = quantel.CSF_CI_space(nmo, n_elec, totspin )
sdrt = space.construct_drt() 
print("testing drt") 
#print(sdrt) 

print("guga utils paths") 
#paths  = csf_basis(a,b,c)
#print(paths)
print("===========") 
print("testing basis") 
basis=space.csf_basis()
count = 0 
for vec in basis: 
    count += 1 
    #print(vec.get_vec()) 
print("guga utils basis size", paths.shape)
print("testing basis size", count) 
#cobj = quantel.Configuration(conf1)
#other_drt = cobj.construct_drt() 
#print("other drt") 
#print(other_drt)   
#print(generate_paldus(conf1)) 
#print("===========") 
#print(cobj.generate_paldus())

#conf3 = np.array([0,3,2,1])
#
#confs = [ conf1, conf2, conf3] 
#
#for conf in confs: 
#    paldus = generate_paldus(conf) 
#    print("guga utils") 
#    print(paldus)
#
#
#    c1 = quantel.Configuration(conf)  
#    c1_paldus = c1.generate_paldus()
#    print("in test") 
#    print(c1_paldus)


#ket=np.array([0,3,0,3,0,0])
#bra=np.array([0,1,0,3,2,0])
#i = 5
#j = 2
bra=np.array([0,3,0,3,0,0])
ket=np.array([0,1,0,3,2,0])
i = 2
j = 5
#
#ket=np.array([0,3,0,3,0,0,3,0])
#bra=np.array([0,3,0,0,3,3,0,0])
#i = 2
#j = 7

#ket=np.array([1,1,1,3,0,1])
#bra=np.array([1,3,2,0,0,1])
#i = 2
#j = 4
braobj = quantel.Configuration((bra))
ketobj = quantel.Configuration((ket))
print("bra", bra )
print("ket", ket)
print("i, j", i, j)


#matrix_element = one_body_matrix_element(bra, ket, i,j)
#print("matrix element", matrix_element)
#matrix_element = testing_one_body_matrix_element(bra, ket, i, j)
#print("test matrix element", matrix_element)
#mb = quantel.MatrixElementCalculator() 
#matrix_element = mb.one_body_coupling(braobj, ketobj, i, j) 
#print("C++ matrix element", matrix_element)
#
#
#print("testing 22 : ", testing_one_body_matrix_element(bra, ket, 2, 2)) 
#print("testing 66 : ", testing_one_body_matrix_element(bra, ket, 6, 6)) 
#print("C++ one body 22: ", mb.one_body_coupling(braobj, ketobj, 2,2) ) 
#
#i,j,k,l = 2, 2, 2, 5
#print("Benchmark two body: ", two_body_matrix_element(bra, ket, i,j,k,l) ) 
#print("C++ two body: ", mb.two_body_coupling(braobj, ketobj, i,j,k,l) ) 
