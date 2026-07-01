import quantel 
import numpy as np
from guga_utils import * 
from quantel.utils.linalg import matrix_print 

conf = [3,3,1,2,1,2,1,1,0,0] 
print("Configuration", conf) 
print("==================") 
#conf = [3,3,1,2,1,0,0,1,1,1,2,2,1,2,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,0,0,0,0] 
n_elec = conf.count(1) + conf.count(2) + 2*conf.count(3)
nmo = len(conf) 
totspin = 0.5*(conf.count(1)-conf.count(2))
a = int(0.5*n_elec - totspin)  
b = int(2*totspin) 
c = int(nmo-a-b) 

###
ref_config = quantel.Configuration(conf)
ci = quantel.CSF_CIspace(nmo, n_elec, totspin )
basis = ci.get_fci_basis()
drt = ci.m_drtobj.get_drt()  
mb = quantel.MatrixElementCalculator() 
print("================") 
print("Two body testing...") 
for i in range(1,len(conf)): 
    for j in range(1,len(conf)): 
        for k in range(1,len(conf)): 
            for l in range(1,len(conf)): 
                print("i,j,k,l", i,j,k,l) 
                Epqrs = quantel.Epphh(i,k,j,l) 
                saved = [ ]
                for obj in basis: 
                    #matrix_element = mb.one_body_coupling(obj, ref_config, Epq)
                    matrix_element = mb.two_body_coupling(obj, ref_config, Epqrs)
                    if np.abs(matrix_element) > 1e-8:
                        saved.append((obj.get_vec(), matrix_element))
                ####
                tsaved = [ ]
                selected_basis = two_body_selected_csf_basis(drt,conf,i,j,k,l)
                for select in selected_basis: 
                    print("select basis function: ", select) 
                    obj = quantel.Configuration(select) 
                    #matrix_element = mb.one_body_coupling(obj, ref_config, Epq)
                    matrix_element = mb.two_body_coupling(obj, ref_config, Epqrs)
                    if np.abs(matrix_element) > 1e-8: 
                        tsaved.append((obj.get_vec(), matrix_element))
                ###
                temporary = ci.m_drtobj.apply_excitation(ref_config, Epqrs) 
                ###
                saved_sorted = sorted(saved, key=lambda x: (tuple(x[0]), x[1]))
                tsaved_sorted = sorted(tsaved, key=lambda x: (tuple(x[0]), x[1]))
                last_sorted =   sorted(temporary, key=lambda x: (tuple(x[0]), x[1]))
                ## 
                if all(tuple(a[0]) == tuple(b[0]) and np.abs(a[1]-b[1]) < 1e-8 
                     for a, b in zip(saved_sorted, saved_sorted)):
                    print("Get the same between benchmarks results, good") 
                else: 
                    print("Benchmarks methods disagree!!!!!!!1") 
                    print(saved) 
                    print(tsaved)
                    break 
                ## 
                if all(tuple(a[0]) == tuple(b[0]) and np.abs(a[1]-b[1]) < 1e-8 
                     for a, b in zip(saved_sorted, last_sorted)):
                    print("Get the same with new good") 
                else: 
                    print("New code disagrees methods disagree!!!!!!!1") 
                    print(saved) 
                    print(last_sorted)
                    break 
 
                print("=============") 

print("=============================")      
print("Print ref Configuration", conf) 


print("==================") 
print("One body testing...") 
for p in range(1,len(conf)): 
    for q in range(1,len(conf)): 
        Epq = quantel.Eph(p,q) 
        print("p,q", p,q) 
        saved = [ ]
        for obj in basis: 
            matrix_element = mb.one_body_coupling(obj, ref_config, Epq)
            if np.abs(matrix_element) > 1e-8:
                saved.append((obj.get_vec(), matrix_element))
        ####
        tsaved = [ ]
        selected_basis = one_body_selected_csf_basis(drt,conf,p,q)
        for select in selected_basis: 
            print("select basis function: ", select) 
            obj = quantel.Configuration(select) 
            matrix_element = mb.one_body_coupling(obj, ref_config, Epq)
            if np.abs(matrix_element) > 1e-8: 
                tsaved.append((obj.get_vec(), matrix_element))
        ###
        temporary = ci.m_drtobj.apply_excitation(ref_config, Epq) 
        ##
        saved_sorted = sorted(saved, key=lambda x: (tuple(x[0]), x[1]))
        tsaved_sorted = sorted(tsaved, key=lambda x: (tuple(x[0]), x[1]))
        last_sorted = sorted(temporary, key=lambda x: (tuple(x[0]), x[1]))
        
        if all(tuple(a[0]) == tuple(b[0]) and np.abs(a[1]-b[1]) < 1e-8 
             for a, b in zip(saved_sorted, saved_sorted)):
            print("Get the same between benchmarks results, good") 
        else: 
            print("Benchmarks methods disagree!!!!!!!1") 
            print(saved) 
            print(tsaved)
            break 
        ## 
        if all(tuple(a[0]) == tuple(b[0]) and np.abs(a[1]-b[1]) < 1e-8 
             for a, b in zip(saved_sorted, last_sorted)):
            print("Get the same with new good") 
        else: 
            print("New code disagrees methods disagree!!!!!!!1") 
            print(saved) 
            print(last_sorted)
            break 
        print("=============") 
