import numpy as np 
import ob_table_array as ob 
import tb_table_array as tb 
# Define unit step Paldus vectors 
step_vectors = np.zeros((4,3), dtype=int)
step_vectors[0,:] = [0,0,1]
step_vectors[1,:] = [0,1,0]
step_vectors[2,:] = [1,-1,1]
step_vectors[3,:] = [1,0,0]

def generate_paldus(u_vec): 
    paldus = np.zeros((len(u_vec)+1,3), dtype=int)
    vec = np.array([0,0,0]) 
    for i in range(len(u_vec)):
        vec += step_vectors[u_vec[i]]
        paldus[i+1,:] = vec 
    return paldus 
 
def pop_vector(bra):
    # bra is a vector of steps
    pops = np.zeros(bra.shape,dtype=int)
    for ind,i in enumerate(bra): 
        if i == 0: 
            pops[ind] = 2 
        elif i==1 or i==2: 
            pops[ind] = 1
        elif i==3: 
            pops[ind] = 0 
    return pops 


## this is a new get drt
# we should probably order these temporary solutions by lexical index as well. 
# so this can be done in the C++ by overwriting the binaries shouldnt be a problem 
def get_drt(a,b,c): 
    # Extract Parameters
    n = a + b + c 
    N = 2*a + b
    S = 0.5*b 
    print("inside drt: ", a, b, c)
    # Create solutions [a,b,c, level_ind, (upwards_connecting) d0, d1, d2, d3 ] 
    soln_array = [[a,b,c,n,0,0,0,0]] 
    tracking_solns = [0] 
    for i in range(n-1,-1,-1):
        start = len(soln_array) 
        temp = [] 
        temp_solns = [] 
        for prev_ind in tracking_solns:
            for d in range(4): 
                ## connect by one of the step vectors
                current = [ soln_array[prev_ind][x] - step_vectors[d,x] for x in range(3) ] 
                current.append(i) 
                if current[0] >= 0 and current[1] >= 0 and current[2] >= 0 :
                    inlist = False
                    for idx, row in enumerate(temp_solns): 
                        if row[:4] == current: 
                            temp_solns[idx][4+d] = prev_ind+1
                            inlist = True 
                    if not inlist:  
                        current += [0,0,0,0]  
                        current[4+d] = prev_ind + 1      
                        temp_solns.append(current)
                        temp.append( start ) 
                        start += 1
                         
        for sol in temp_solns: 
            soln_array.append(sol) 
        tracking_solns = temp  
    return np.array(soln_array, dtype=int) 
## so this gives the DRT and the coupling table in one long array 

def csf_basis(a,b,c):
    drt = get_drt(a,b,c) 
    nmo = a+b+c  
    spawn = [ [], drt.shape[0] ] #is the lexical name      
    configurations = [spawn] # so the first solutions? 
    # Subsequent steps 
    #iterate over the levels 
    for level in range(0, nmo):
        new_configurations = [] 
        # Get the current level node names 
        # these are the lexical names -1
        lexical_inds = np.argwhere(drt[:,3] == level).reshape(-1)  
        for index in lexical_inds:
            steps = np.argwhere(drt[index,4:]!=0).reshape(-1)  
            # lexical index and node that are connected to current solution 
            for configuration in configurations: 
                # so checking for our starting point 
                if configuration[1]==index+1:
                    #new do the current steps 
                    for step in steps: 
                        new_configurations.append( [configuration[0]+[step], drt[index,4:][step] ] )
        configurations = new_configurations 
    
    ####
    order = [] 
    for configuration in configurations: 
        v = 0 
        for aind, a in enumerate(configuration[0]): 
            v+= a*(10**aind) 
        order.append(v) 
    order = np.argsort(order)
    configurations =[configurations[i] for i in order]
    
    paths = np.zeros((len(configurations),nmo), dtype=int)
    for ind, path in enumerate(configurations): 
        if path[1]==1:
            paths[ind,:] = np.array(path[0]) 
        else: 
            print(f"path {ind} did not converge" )  
    return paths 



## so here we can use the DRT to find all the possible CSFs that are connected - but we can place additional constraints on which CSFs we pick 
# for the one body terms we should restrict the terms to only picking the paths which have the same number of unpaired electrons
# or actually more importantly that we can only flip the number 1 or 2 - we cannot flip a 0 or 3. 

## dont need coupling table as it is present inside the new drt! And its better as well!  
#def coupling_table(single_solns, key): 
#    upwards_connecting = np.zeros((len(single_solns)-1, 4), dtype=int)
#    downwards_connecting = np.zeros((len(single_solns)-1, 4), dtype=int)
#    for i in key[1,1:][::-1]: 
#        # current solution index 
#        sind = i - 1
#        level = key[0,sind]
#        # get all solutions in above level 
#        indices = np.array(key[0,:] == level+1)
#        levelplus_solns = [single_solns[indices][i,:] for i in range(single_solns[indices].shape[0]) ] 
#        # Can definitely write this in a nicer way - 3 nested for loops is a bit excessive 
#        for u in range(4):
#            # Calc potential solution
#            # so adding the steps to the current solution 
#            comb = [ step_vectors[u,a]+single_solns[sind][a] for a in range(3)]   
#            # See if its inside the above solutions
#            for lpind, lp_soln in enumerate(levelplus_solns): 
#                if comb==list(lp_soln): 
#                    tname = key[1,:][indices][lpind] 
#                    # so then upwards connect saves tname in the lower solution index?
#                    # am slightly confused why is this sind -1 ?  
#                    upwards_connecting[sind - 1, u] = tname 
#                    downwards_connecting[ tname - 1   , u] = sind+1  
#    ## 
#    # so the connecting tables are what exactly
#    # So they are upwards connecting - with 4 columns saying which solution connects to 
#    # but it shouldnt be giv
#    # upwards_connecting[ sol, : ] = 
#    return upwards_connecting, downwards_connecting  


# extract step searches through the drt and finds the other solutions which the graph node connects to. 
def extract_step(graph_node, upwards_connecting):
    # np.where(upwards_connecting[graph_node-2,:] != 0 ) # this tells us where the solutions maybe be connected. 
    # i dont think we really need a whole other function for this? 
    
    # so this tells us which lexical index are connected to the current solution 
    step_inds = np.array(np.where(upwards_connecting[graph_node-2,:] !=0)).reshape((-1))   
    # then we find these nodes 
    new_nodes = np.array(upwards_connecting[graph_node-2,:][step_inds]).reshape((-1))
    return step_inds, new_nodes 


# iterates through the drt to construct the complete CSF basis.
# graph was the drt 
# key was the level index 
# upwards connecting was the lexical index I guess
# this can all be put into one drt_array 
 
#def csf_basis(graph, key, upwards_connecting): 
#    n = np.sum(graph[0]) # number of orbtials 
#    spawn = [ [], len(graph) ] # is this a solution index?      
#    csfs = [spawn] # so the first solutions? 
#    # Subsequent steps 
#    #iterate over the levels 
#    for level in range(0, n):
#        new_csfs = [] 
#        # Get the current level node names 
#        indices = np.array(key[0,:] == level)
#        # are these really the level plus names? 
#        levelplus_names = key[1,:][indices] 
#        #Iterate over these names 
#        # leve
#        for name in levelplus_names:
#            # lexical index and node that are connected to current solution 
#            step_inds, new_nodes = extract_step(name, upwards_connecting)
#            for csf in csfs: 
#                if csf[1]==name:
#                    for a in range(len(new_nodes)): 
#                        new_csfs.append( [csf[0]+[int(step_inds[a])], int(new_nodes[a]) ] )
#        csfs = new_csfs 
#    order = [] 
#    for csf in csfs: 
#        v = 0 
#        for aind, a in enumerate(csf[0]): 
#            v+= a*(10**aind) 
#        order.append(v) 
#    order = np.argsort(order)
#    csfs =[csfs[i] for i in order]
#    
#    paths = np.zeros((len(csfs),n), dtype=int)
#    for ind, path in enumerate(csfs): 
#        if path[1]==1:
#            paths[ind,:] = np.array(path[0]) 
#        else: 
#            print(f"path {ind} did not converge" )  
#    return paths 

def one_body_matrix_element(bra, ket, i, j): 
    # <bra | Eij | ket > 
    """ 
    First checks 
    1. same number of electrons, spatial orbitals
    2. same S 
    """ 
    bra_paldus = generate_paldus(bra)    
    ket_paldus = generate_paldus(ket)
    
    if not np.array_equal(bra_paldus[-1,:], ket_paldus[-1,:]): 
        print(" Different N, n or S values")
        print("ket", ket_paldus[-1,:])
        print("bra", bra_paldus[-1,:])
        return 0.0 
    n = np.sum(bra_paldus[-1,:])

    # Extract indices in the range of excitation generator 
    S0 = [ a for a in range(min(i,j), max(i,j)+1) ]
    # Define loop head and loop tail index
    outside_loop = [ a for a in range(S0[0]) ] 
    outside_loop += [ a for a in range(S0[-1], n+1) ] 

    for level in outside_loop: 
        if not np.array_equal(bra_paldus[level,:],ket_paldus[level,:]):
            #print("Outside loop paths are not the same") 
            return 0.0 
    
    # Calculate Loop values 
    # Diagonal excitation Eii 
    if i==j:
        return ob.table_one[bra[i-1],ket[i-1], 0]( ket_paldus[i,1] )            
    
    # Off-diagonal Excitation Eij 
    # Check if Raising or Lowering operator
    # R = 0 , L = 1 
    RorL = int(i - j > 0) 
    
    # Iterate over levels in loop 
    matrix_element = 1.0
    for k in S0:
        if matrix_element == 0.0:
            return 0.0 
        
        d1 = bra[k-1]
        d2 = ket[k-1]
        b = ket_paldus[k,1]
        Deltab = ket_paldus[k,1] - bra_paldus[k,1]
        
        #if k==S0[-1]:
        #    # Loop head  
        #    matrix_element *= ob.table_one[d1,d2,RorL + 1 ](b)
        #else: 
        #    if int(np.abs(Deltab))!=1: 
        #       return 0.0  
        #    
        #    #print("Deltab", Deltab)
        #    #print("-----")
        #    if k==S0[0]:
        #        # Loop tail
        #        matrix_element *= ob.table_one[d1,d2,RorL + 3](b)
        #
        #    else: 
        #        Dind = 0 if Deltab == -1 else 1
        #        #print("Dind", Dind) 
        #        #print("-------") 
        #        matrix_element *= ob.table_two[d1,d2,RorL, Dind](b) 
        
        if k==S0[-1]:
            # Loop head  
            matrix_element *= ob.table_one[d1,d2,RorL + 1 ](b)
        elif int(np.abs(Deltab))!=1: 
            return 0.0  
            
        elif k==S0[0]:
            # Loop tail
            matrix_element *= ob.table_one[d1,d2,RorL + 3](b)
        
        else: 
            Dind = 0 if Deltab == -1 else 1
            #print("Dind", Dind) 
            #print("-------") 
            matrix_element *= ob.table_two[d1,d2,RorL, Dind](b) 
    return matrix_element



def one_body_fragment(level,d1,d2,b, deltab, S, RorL): 
    if level==S[-1]:
        # Loop head  
        factor = ob.table_one[d1,d2,RorL + 1 ](b)
    elif int(np.abs(deltab))!=1: 
        factor =  0.0   
    elif level==S[0]:
        # Loop tail
        factor = ob.table_one[d1,d2,RorL + 3](b) 
    else: 
        Dind = 0 if deltab == -1 else 1
        factor = ob.table_two[d1,d2,RorL, Dind](b) 
    return factor 

def testing_one_body_matrix_element(bra, ket, i, j): 
    # <bra | Eij | ket > 
    """ 
    First checks 
    1. same number of electrons, spatial orbitals
    2. same S 
    """ 
    bra_paldus = generate_paldus(bra)    
    ket_paldus = generate_paldus(ket)
    
    if not np.array_equal(bra_paldus[-1,:], ket_paldus[-1,:]): 
        print(" Different N, n or S values")
        print("ket", ket_paldus[-1,:])
        print("bra", bra_paldus[-1,:])
        return 0.0 
    n = np.sum(bra_paldus[-1,:])

    # Extract indices in the range of excitation generator 
    S0 = [ a for a in range(min(i,j), max(i,j)+1) ]
    # Define loop head and loop tail index
    outside_loop = [ a for a in range(S0[0]) ] 
    outside_loop += [ a for a in range(S0[-1], n+1) ] 

    for level in outside_loop: 
        if not np.array_equal(bra_paldus[level,:],ket_paldus[level,:]):
            #print("Outside loop paths are not the same") 
            return 0.0 
    
    # Calculate Loop values 
    # Diagonal excitation Eii 
    if i==j:
        return ob.table_one[bra[i-1],ket[i-1], 0]( ket_paldus[i,1] )            
    
    # Off-diagonal Excitation Eij 
    # Check if Raising or Lowering operator
    # R = 0 , L = 1 
    RorL = int(i - j > 0) 
    
    # Iterate over levels in loop 
    matrix_element = 1.0
    for k in S0:
        if matrix_element == 0.0:
            return 0.0 
        
        d1 = bra[k-1]
        d2 = ket[k-1]
        b = ket_paldus[k,1]
        deltab = ket_paldus[k,1] - bra_paldus[k,1]
        matrix_element *= one_body_fragment(k, d1, d2, b, deltab, S0, RorL) 
    
    return matrix_element

def resolve_two_body_matrix_element( bra, ket, i, j, k, l):
    # <bra| eij,kl |ket> = Sum [ <bra| Eij|m><m| Ekl | ket ] - Delta(j,k) <bra| Eil | ket>  
    bra_paldus = generate_paldus(bra)    
    ket_paldus = generate_paldus(ket)
    if not np.array_equal(bra_paldus[-1,:], ket_paldus[-1,:]): 
        print("Different N, n or S values")
        return 0.0 
    
    n = np.sum(bra_paldus[-1,:])
    # Check paths outside loop  
    true_loop = set(range(min(i,j,k,l), max(i,j,k,l)+1))
    out_of_true_loop = set(range(bra.shape[0]+1)) - true_loop 
    for level in list(out_of_true_loop):
        if not np.array_equal(bra_paldus[level,:],ket_paldus[level,:]):
            return 0.0 
    
    matrix_element = 0.0 
    # One-body contribution
    if j==k:
        matrix_element -= one_body_matrix_element( bra, ket, i,l) 
        #print("one body correction",matrix_element)
    
    basis =  csf_basis(bra_paldus[-1,0],bra_paldus[-1,1],bra_paldus[-1,2])
    for ind in range(basis.shape[0]):           
        matrix_element += one_body_matrix_element(bra, basis[ind,:], i, j) * one_body_matrix_element(basis[ind,:], ket, k, l) 
    return matrix_element

####

def convertRorL(string): 
    if string=="R": 
        return 0 
    elif string=="L": 
        return 1

def headortails(ind, indices): 
    if ind == max(indices): 
        return "h" 
    elif ind == min(indices): 
        return "t" 
    else: 
        return "" 

def get_Dind(Deltab): 
    if Deltab==1 or Deltab==0: 
        return 1     
    elif Deltab==-1 or Deltab==-2: 
        return 0     
    elif Deltab==2: 
        return 2

def two_body_matrix_element( bra, ket, i, j, k, l):
    # this is in chemists notation 
    # <bra| eij,kl |ket> = Sum [ <bra| Eij|m><m| Ekl | ket ] - Delta(j,k) <bra| Eil | ket>  
    bra_paldus = generate_paldus(bra)    
    ket_paldus = generate_paldus(ket)
    if not np.array_equal(bra_paldus[-1,:], ket_paldus[-1,:]): 
        print("Different N, n or S values")
        return 0.0 
    
    n = np.sum(bra_paldus[-1,:])
    # Check paths outside loop  
    true_loop = set(range(min(i,j,k,l), max(i,j,k,l)+1))
    out_of_true_loop = set(range(bra.shape[0]+1)) - true_loop 
    for level in list(out_of_true_loop):
        if not np.array_equal(bra_paldus[level,:],ket_paldus[level,:]):
            return 0.0 
    
    #print("Paths don't diverge")
    #print("i,j,k,l",i,j,k,l)
    seta = set(range(min(i,j), max(i,j)+1)) 
    setb = set(range(min(k,l), max(k,l)+1))
    
    # Need to know if they are R, L or D     
    if i - j > 0: 
        a_class = "L" 
    elif i - j < 0: 
        a_class = "R" 
    else: 
        a_class = "D" 
    if k - l > 0: 
        b_class = "L" 
    elif k - l < 0: 
        b_class = "R" 
    else: 
        b_class = "D"
    ab_classes = [ a_class, b_class ] 
    #print("ab_classes", ab_classes)  
    
    ## tells us which are diagonal if any 
    #diags = [ ind for ind, value in enumerate(ab_classes) if value =="D" ] 
    ##print("diags", diags)  
    #if len(diags)==2:
    #    #print("Inside diags len 2")  
    #    # One diagonal element is nonzero < bra |e_ijkl| bra > 
    #    if np.allclose(bra,ket):  
    #        matrix_element =  ob.table_one[bra[i-1],bra[i-1], 0]( bra_paldus[i,1] )*ob.table_one[bra[k-1],bra[k-1], 0]( bra_paldus[k,1] ) 
    #        if j==k: 
    #            matrix_element -= ob.table_one[bra[i-1],bra[i-1], 0]( bra_paldus[i,1] )
    #        return matrix_element
    #    else:
    #        #print("Zero by off diag") 
    #        return 0.0  
    #
    #elif len(diags)==1: 
    #    #print("Inside diags len 1")  
    #    # Example: Pop.(i) in bra * <bra| Ekl | ket > (resolution of identity with Eii) 
    #    if k == l :
    #        matrix_element =  one_body_matrix_element(bra, ket, i, j)*ob.table_one[ket[k-1],ket[k-1], 0]( ket_paldus[k,1] )
    #    elif i == j : 
    #        # Pop.(i) in bra * <bra| Ekl | ket > (resolution of identity with Eii) 
    #        matrix_element =  ob.table_one[bra[i-1],bra[i-1], 0]( bra_paldus[i,1] )*one_body_matrix_element(bra, ket, k, l)  
    #    if j == k:
    #            print("matrix element here ", matrix_element)  
    #            matrix_element -= one_body_matrix_element(bra,ket, i,l) 
    #    return matrix_element 

    matrix_element = 0.0 
    if (i==j or j==k): 
        if (i==j and k==l) : 
            matrix_element += one_body_matrix_element(bra, ket, i, j)*one_body_matrix_element(bra, ket, k, l)
        elif (i==j): 
            matrix_element += one_body_matrix_element(bra, bra, i, j)*one_body_matrix_element(bra, ket, k, l)
        elif (k==l): 
            matrix_element += one_body_matrix_element(bra, ket, i, j)*one_body_matrix_element(ket, ket, k, l)

        if j==k: 
            matrix_element -= one_body_matrix_element(bra, ket, i, l)
        return matrix_element 

    S1 = seta & setb 
    S2 = seta.union(setb) - S1 
    print("i,j", i,j)
    print("k,l", k,l)
    print("S1", S1)
    print("S2", S2)
    # 
    RorLs = [ convertRorL(a_class), convertRorL(b_class) ]
    matrix_element = 1.00  
    # Non-overlapping range
    for ind in list(S2):
        print("in S2", ind) 
        # Check if its an R or L operator 
        if ind in seta: 
            RorL = RorLs[0]
            HorT = headortails(ind, list(seta) ) 
        elif ind in setb: 
            RorL = RorLs[1]
            HorT = headortails(ind, list(setb) ) 
        else: 
            print("Error") 
            return
        print("HorT", HorT) 
        if matrix_element == 0.0:
            print("Zero by non overlapping") 
            return 0.0 
        
        d1 = bra[ind-1]
        d2 = ket[ind-1]
        print("ket step", d2) 
        b = ket_paldus[ind,1]
        Deltab = ket_paldus[ind,1] - bra_paldus[ind,1]
        
        if HorT=="h":
            # Loop head
            matrix_element *= ob.table_one[d1,d2,RorL + 1 ](b)
        else: 
            if int(np.abs(Deltab))!=1:
                matrix_element = 0.0  
                print("Zero by non overlapping 2") 
                print("Zero by non overlapping 2")
                print("matrix element", matrix_element)  
                return matrix_element 
            
            if HorT=="t":
                # Loop tail
                matrix_element *= ob.table_one[d1,d2,RorL + 3](b)
        
            else: 
                Dind = 0 if Deltab == -1 else 1
                matrix_element *= ob.table_two[d1,d2,RorL, Dind](b) 
    
    # Check non zero before doing two body section     
    if matrix_element == 0.0:
        print("Zero by non overlapping 3") 
        return matrix_element 
   

    # Overlapping range 
    x0 = 1.0    
    x1 = 1.0
    #print("S1 now")  
    for ind in list(S1): 
        #print("matrix element", matrix_element) 
        if matrix_element == 0.0:
            #print("Zero by overlapping") 
            return 0.0 
        op_classes = []
        for a, reference_set in enumerate([ seta, setb ]): 
            op_classes.append( headortails(ind, list(reference_set)) + ab_classes[a] ) 
        # so this is returning if we are in headortails in either set a or set b 
        #        

        #print(op_classes) 
        d1 = bra[ind-1]
        d2 = ket[ind-1]
        b = ket_paldus[ind,1]
        Deltab = ket_paldus[ind,1] - bra_paldus[ind,1]
        if (op_classes==["hR", "hR" ]) or (op_classes==["tL","tL"]):
            x0 *= tb.table_one[d1, d2, 0 , 0 ]( b )  
            x1 *= tb.table_one[d1, d2, 0 , 1 ]( b )  
        
        elif (op_classes==["tR", "tR" ]) or (op_classes==["hL","hL"]):
            x0 *= tb.table_one[d1, d2, 1 , 0 ]( b )  
            x1 *= tb.table_one[d1, d2, 1 , 1 ]( b )  
        elif ( ("hR" in op_classes) and ("hL" in op_classes) ) :
            x0 *= tb.table_one[d1, d2, 2 , 0 ]( b )  
            x1 *= tb.table_one[d1, d2, 2 , 1 ]( b )  
        elif ( ("tR" in op_classes) and ("tL" in op_classes) ) :
            x0 *= tb.table_one[d1, d2, 3 , 0 ]( b )  
            x1 *= tb.table_one[d1, d2, 3 , 1 ]( b )  
        
        elif ( ("tR" in op_classes) and ("hR" in op_classes) ):
            if int(np.abs(Deltab)) != 1: 
                return 0.0 
            matrix_element *= tb.table_two[d1, d2, 0 , get_Dind(Deltab)]( b )  
        elif ( ("tL" in op_classes) and ("hL" in op_classes) ):
            if int(np.abs(Deltab)) != 1: 
                return 0.0 
            matrix_element *= tb.table_two[d1, d2, 1 , get_Dind(Deltab)]( b )  
        elif ( ("hR" in op_classes) and ("tL" in op_classes) ):
            if int(np.abs(Deltab)) != 1: 
                return 0.0 
            matrix_element *= tb.table_two[d1, d2, 2 , get_Dind(Deltab)]( b )  
        elif ( ("tR" in op_classes) and ("hL" in op_classes) ):
            if int(np.abs(Deltab)) != 1: 
                return 0.0 
            matrix_element *= tb.table_two[d1, d2, 3 , get_Dind(Deltab)]( b )  

           
        elif op_classes==["R","hR"]:
            if int(np.abs(Deltab)) != 1: 
                return 0.0
            x0 *= tb.table_three[d1, d2, 0 , get_Dind(Deltab), 0]( b )  
            x1 *= tb.table_three[d1, d2, 0 , get_Dind(Deltab), 1]( b )  
            #print("x0", tb.table_three[d1, d2, 0 , get_Dind(Deltab), 0]( b ))  
            #print("x1", tb.table_three[d1, d2, 0 , get_Dind(Deltab), 1]( b ))  
        elif op_classes==["hR","R"]:
            if int(np.abs(Deltab)) != 1: 
                return 0.0 
            x0 *= tb.table_three[d1, d2, 1 , get_Dind(Deltab), 0]( b )  
            x1 *= tb.table_three[d1, d2, 1 , get_Dind(Deltab), 1]( b )  
            #print("x0", tb.table_three[d1, d2, 1 , get_Dind(Deltab), 0]( b ))  
            #print("x1", tb.table_three[d1, d2, 1 , get_Dind(Deltab), 1]( b ))  
        elif op_classes==["hL","L"]:
            if int(np.abs(Deltab)) != 1: 
                return 0.0 
            x0 *= tb.table_three[d1, d2, 2 , get_Dind(Deltab), 0]( b )  
            x1 *= tb.table_three[d1, d2, 2 , get_Dind(Deltab), 1]( b )  
        elif op_classes==["L","hL"]:
            if int(np.abs(Deltab)) != 1: 
                return 0.0 
            x0 *= tb.table_three[d1, d2, 3 , get_Dind(Deltab), 0]( b )  
            x1 *= tb.table_three[d1, d2, 3 , get_Dind(Deltab), 1]( b )  
        elif ( ("hR" in op_classes) and ("L" in op_classes) ):
            if int(np.abs(Deltab)) != 1: 
                return 0.0 
            x0 *= tb.table_three[d1, d2, 4 , get_Dind(Deltab), 0]( b )  
            x1 *= tb.table_three[d1, d2, 4 , get_Dind(Deltab), 1]( b )  
        elif ( ("R" in op_classes) and ("hL" in op_classes) ):
            if int(np.abs(Deltab)) != 1: 
                return 0.0 
            x0 *= tb.table_three[d1, d2, 5 , get_Dind(Deltab), 0]( b )  
            x1 *= tb.table_three[d1, d2, 5 , get_Dind(Deltab), 1]( b )  

        elif op_classes==["tR","R"]:
            if int(np.abs(Deltab)) != 2 and int(np.abs(Deltab)) != 0: 
                return 0.0 
            x0 *= tb.table_four[d1, d2, 0 , get_Dind(Deltab), 0]( b )  
            x1 *= tb.table_four[d1, d2, 0 , get_Dind(Deltab), 1]( b )  
            #print("d1, d2", d1, d2)
            #print("db", Deltab)
            #print("b", b)
            #print("x0", tb.table_four[d1, d2, 0 , get_Dind(Deltab), 0]( b ))  
            #print("x1", tb.table_four[d1, d2, 0 , get_Dind(Deltab), 1]( b ))  
        elif op_classes==["R","tR"]:
            if int(np.abs(Deltab)) != 2 and int(np.abs(Deltab)) != 0: 
                return 0.0 
            x0 *= tb.table_four[d1, d2, 1 , get_Dind(Deltab), 0]( b )  
            x1 *= tb.table_four[d1, d2, 1 , get_Dind(Deltab), 1]( b )  
            #print("d1, d2", d1, d2)
            #print("db", Deltab)
            #print("b", b)
            #print("x0", tb.table_four[d1, d2, 1 , get_Dind(Deltab), 0]( b ))  
            #print("x1", tb.table_four[d1, d2, 1 , get_Dind(Deltab), 1]( b ))  
        elif op_classes==["L","tL"]:
            if int(np.abs(Deltab)) != 2 and int(np.abs(Deltab)) != 0: 
                return 0.0 
            x0 *= tb.table_four[d1, d2, 2 , get_Dind(Deltab), 0]( b )  
            x1 *= tb.table_four[d1, d2, 2 , get_Dind(Deltab), 1]( b )  
        elif op_classes==["tL","L"]:
            if int(np.abs(Deltab)) != 2 and int(np.abs(Deltab)) != 0: 
                return 0.0 
            x0 *= tb.table_four[d1, d2, 3 , get_Dind(Deltab), 0]( b )  
            x1 *= tb.table_four[d1, d2, 3 , get_Dind(Deltab), 1]( b )  
        elif op_classes==["R","R"]:
            if int(np.abs(Deltab)) != 2 and int(np.abs(Deltab)) != 0: 
                return 0.0 
            x0 *= tb.table_four[d1, d2, 4 , get_Dind(Deltab), 0]( b )  
            x1 *= tb.table_four[d1, d2, 4 , get_Dind(Deltab), 1]( b )  
        elif op_classes==["L","L"]:
            if int(np.abs(Deltab)) != 2 and int(np.abs(Deltab)) != 0: 
                return 0.0 
            x0 *= tb.table_four[d1, d2, 5 , get_Dind(Deltab), 0]( b )  
            x1 *= tb.table_four[d1, d2, 5 , get_Dind(Deltab), 1]( b )  
        elif ( ("R" in op_classes) and ("tL" in op_classes) ):
            if int(np.abs(Deltab)) != 2 and int(np.abs(Deltab)) != 0: 
                return 0.0 
            #print("Deltab", Deltab )  
            #print("Dind", get_Dind(Deltab) )  
            x0 *= tb.table_four[d1, d2, 6 , get_Dind(Deltab), 0]( b )  
            x1 *= tb.table_four[d1, d2, 6 , get_Dind(Deltab), 1]( b )  
        elif ( ("tR" in op_classes) and ("L" in op_classes) ):
            if int(np.abs(Deltab)) != 2 and int(np.abs(Deltab)) != 0: 
                return 0.0
            x0 *= tb.table_four[d1, d2, 7 , get_Dind(Deltab), 0]( b )  
            x1 *= tb.table_four[d1, d2, 7 , get_Dind(Deltab), 1]( b )  
        elif ( ("R" in op_classes) and ("L" in op_classes) ):
            if int(np.abs(Deltab)) != 2 and int(np.abs(Deltab)) != 0: 
                return 0.0 
            x0 *= tb.table_four[d1, d2, 8 , get_Dind(Deltab), 0]( b )  
            x1 *= tb.table_four[d1, d2, 8 , get_Dind(Deltab), 1]( b )  
        else: 
            print("Fell through Error", op_classes)
    #print("final x0", x0) 
    #print("final x1", x1)
    #print("combo", x0 +x1 )   
    matrix_element *= ( x0 + x1 ) 
    return matrix_element 

######
#### Need to look at which ones we should even bother calculating 
####### are there any simplification that we can make?
# According to the differences in the populations of the spin orbtials - this population we could easily from the paldus table. 

#
def one_body_coupling(bra,ket,hcore): 
    bra_pop = pop_vector(bra) 
    ket_pop = pop_vector(ket) 
    pop_diff = bra_pop - ket_pop
    
    #if 2 in pop_diff: 
    #     
    #

    matrix_element = 0 
    for i in range(hcore.shape[0]): 
        for j in range(hcore.shape[0]):
            factor = one_body_matrix_element(bra, ket, i+1,j+1)
            term  = hcore[i,j]*factor 
            matrix_element += factor 
    return matrix_element 

#### Need to look at which ones we should even bother calculating 
####### are there any simplification that we can make?
def two_body_coupling(bra,ket,eri): 
    matrix_element = 0 
    for i in range(eri.shape[0]): 
        for j in range(eri.shape[0]):
            for k in range(eri.shape[0]): 
                for l in range(eri.shape[0]):
                    factor = two_body_matrix_element(bra, ket, i+1,j+1,k+1,l+1)
                    term = eri[i,k,j,l]*factor  
                    matrix_element += term
    return matrix_element 

######
def hamiltonian_coupling(bra, ket, obj): 
    hcore_mat = np.linalg.multi_dot((obj.mo_coeff.T, obj.integrals.oei_matrix(), obj.mo_coeff))
    tei_array = obj.integrals.tei_ao_to_mo(obj.mo_coeff, obj.mo_coeff, obj.mo_coeff, obj.mo_coeff, True, False)
    one_body = one_body_coupling(bra, ket, hcore_mat)
    two_body = 0.5*two_body_coupling(bra, ket, tei_array)
    return one_body, two_body  

def old_one_body_selected_csf_basis(drt,ref_configuration,p,q):
    # This finds all the CSFs that have a nonzero matrix element with the one body singlet excitation of | start conf > 
    # Finds all m such that < m | E_pq | start >, where p < q (R operator only) 
    assert p <= q, "Only evaluates D,R operators, p < q"  
    drt = np.array(drt, dtype=int)
    ## Catch the diagonal case: 
    if (p==q):
        return [ref_configuration]  

    # Cheap to construct the full DRT - we will also already have the full DRT so we can keep it! 
    start = np.zeros(3,dtype=int) 
    end = np.zeros(3,dtype=int) 
    # p, q start at 0, -1 for indexing step vector
    for i in range(p-1): 
        start += step_vectors[ ref_configuration[i] ] 
    for i in range(q): 
        end += step_vectors[ ref_configuration[i] ] 
    start_ind = int(np.where((drt[:,:3] == start ).all(axis=1))[0][0]) 
    end_ind = int(np.where((drt[:,:3] == end ).all(axis=1))[0][0])
    nmo = np.sum(drt[0,:3]) 
    
    # Subsequent steps 
    #iterate over the levels
    
    # Tail condition 
    spawn = [ [], int(start_ind+1) ] # lexical name      
    level = drt[start_ind,3]
    lexical_inds = [start_ind]
    #>>>>>>>>>>>>>>>> 
    #ref_step = ref_configuration[level-1]
    #---------------
    ref_step = ref_configuration[level-1]
    #>>>>>>>>>>>>>>>> 
    steps = []
    if ref_step==0:
        steps = np.argwhere(drt[start_ind,4:]!=0).reshape(-1)  
        steps = [ x for x in steps if x in [1,2]] 
    elif ref_step==1 or ref_step==2:
        steps = np.argwhere(drt[start_ind,4:]!=0).reshape(-1)  
        steps = [3] if 3 in steps else []
    
    if len(steps) == 0 :
        return []  
    configurations = [] 
    for step in steps: 
        configurations.append( [ spawn[0]+[step], drt[start_ind,4:][step] ] )
   
    # Loop body 
    for level in range(drt[start_ind,3]+1, drt[end_ind,3]-1):
        ## HAVE MADE THIS BIG CHANGE - I NEED TO TEST IT 
        #ref_step = ref_configuration[level]
        ref_step = ref_configuration[level-1]
        #>>>>>>>>>>
        new_configurations = [] 
        lexical_inds = [] 
        for conf in configurations:
            if conf[1]-1 not in lexical_inds:  
                lexical_inds.append(conf[1]-1) 
        lexical_inds  = np.array(lexical_inds)        
        for index in lexical_inds:
            # lexical index and node that are connected to current solution 
            steps = [] 
            if ref_step==0 or ref_step==3:
                steps = np.argwhere(drt[index,4:]!=0).reshape(-1)  
                steps = [ref_step] if ref_step in steps else []
 
            elif ref_step==1 or ref_step==2:
                steps = np.argwhere(drt[index,4:]!=0).reshape(-1)  
                steps = [ x for x in steps if x in [1,2]] 
           
            ## testing this removal procedure!  
            if len(steps) == 0 :
                configurations = [ x for x in configurations if x[1]!=index+1] 
                continue   
                 
            for configuration in configurations: 
                # so checking for our starting point 
                if configuration[1]==index+1:
                    #new do the current steps 
                    for step in steps: 
                        new_configurations.append( [configuration[0]+[step], drt[index,4:][step] ] )
        configurations = new_configurations 
    
    # Head condition 
    level = drt[end_ind,3]-1 
    #>>>>>>
    #ref_step = ref_configuration[level]
    #-------
    ref_step = ref_configuration[level-1]
    #>>>>>>
    new_configurations = [] 
    lexical_inds = [] 
    for conf in configurations:
        if conf[1]-1 not in lexical_inds:  
            lexical_inds.append(conf[1]-1) 
    lexical_inds  = np.array(lexical_inds)        
    for index in lexical_inds:
        if ref_step==1 or ref_step==2:
            steps = np.argwhere(drt[index,4:]==end_ind+1).reshape(-1) 
            steps = [0] if 0 in steps else []
        elif ref_step==3:
            steps = np.argwhere(drt[index,4:]==end_ind+1).reshape(-1)  
            steps = [ x for x in steps if x in [1,2]]
        else: 
            steps=[]
        if len(steps) == 0 :
            continue 
             
        for configuration in configurations: 
            # so checking for our starting point 
            if configuration[1]==index+1:
                #new do the current steps 
                for step in steps:
                    new_configurations.append( [configuration[0]+[step], drt[index,4:][step] ] )
    configurations = new_configurations 

    # Order the states 
    order = [] 
    for configuration in configurations: 
        v = 0 
        for ind, i in enumerate(configuration[0]): 
            v+= i*(10**ind) 
        order.append(v) 
    order = np.argsort(order)
    configurations =[configurations[i] for i in order]
    paths = np.zeros((len(configurations),nmo), dtype=int)
    for ind, path in enumerate(configurations): 
        if path[1]==end_ind+1:
            paths[ind,:p-1] = ref_configuration[:p-1]
            paths[ind,q:] = ref_configuration[q:]
            paths[ind,p-1:q] = np.array(path[0]) 
        else: 
            print(f"path {ind} did not converge" )  
    return paths 
##################################################
###################################################


def one_body_selected_csf_basis(drt,ref_configuration,p,q):
    # This finds all the CSFs that have a nonzero matrix element with the one body singlet excitation of | start conf > 
    # Finds all m such that < m | E_pq | start >, where p < q (R operator only) 
    #assert p <= q, "Only evaluates D,R operators, p < q"  
    drt = np.array(drt, dtype=int)
    ## Catch the diagonal case: 
    if (p==q):
        return [ref_configuration]  
    tail = min(p,q) 
    head = max(p,q) 

    # Cheap to construct the full DRT - we will also already have the full DRT so we can keep it! 
    start = np.zeros(3,dtype=int) 
    end = np.zeros(3,dtype=int) 
    # p, q start at 0, -1 for indexing step vector
    for i in range(tail-1): 
        start += step_vectors[ ref_configuration[i] ] 
    for i in range(head): 
        end += step_vectors[ ref_configuration[i] ] 
    start_ind = int(np.where((drt[:,:3] == start ).all(axis=1))[0][0]) 
    end_ind = int(np.where((drt[:,:3] == end ).all(axis=1))[0][0])
    nmo = np.sum(drt[0,:3]) 
    
    # Subsequent steps 
    #iterate over the levels
    
    # Tail condition 
    spawn = [ [], int(start_ind+1) ] # lexical name      
    configurations = [ spawn ]  
    # Loop body 
    for level in range(drt[start_ind,3], drt[end_ind,3]):
        ref_step = ref_configuration[level-1]
        allowed_steps = one_body_drt_step(ref_step, level, p,q)  
        new_configurations = [] 
        lexical_inds = [] 
        for conf in configurations:
            if conf[1]-1 not in lexical_inds:  
                lexical_inds.append(conf[1]-1) 
        lexical_inds  = np.array(lexical_inds)        
        for index in lexical_inds:
            if level == head - 1:
                steps = np.argwhere(drt[index,4:]==end_ind+1).reshape(-1)
            else: 
                steps = np.argwhere(drt[index,4:]!=0).reshape(-1)
         
            steps = [ x for x in steps if x in allowed_steps] 
          

            ## Wait do I need this? Surely because its not going to kept with new configurations
            #if len(steps) == 0 :
            #    # Remove all configurations at this lexical node as steps = [ ]  
            #    configurations = [ x for x in configurations if x[1]!=index+1 ] 
            #else: 
            #    for step in steps: 
            #        new_configurations.append( [configuration[0]+[step], drt[index,4+step] ] )
            if len(steps) != 0: 
                for configuration in configurations: 
                    if configuration[1]==index+1: 
                        for step in steps: 
                            new_configurations.append( [configuration[0]+[step], drt[index,4+step] ] )
                     
        configurations = new_configurations 
        #print(configurations)  
    
    print("final configurations") 
    print(configurations)  
    # Order the states 
    order = [] 
    for configuration in configurations: 
        v = 0 
        for ind, i in enumerate(configuration[0]): 
            v+= i*(10**ind) 
        order.append(v) 
    order = np.argsort(order)
    configurations =[configurations[i] for i in order]
    paths = np.zeros((len(configurations),nmo), dtype=int)
    for ind, path in enumerate(configurations): 
        if path[1]==end_ind+1:
            paths[ind,:tail-1] = ref_configuration[:tail-1]
            paths[ind,head:] = ref_configuration[head:]
            paths[ind,tail-1:head] = np.array(path[0]) 
        else: 
            print(f"path {ind} did not converge" )  
    return paths 
##################################################
###################################################

def two_body_selected_csf_basis(drt,ref_configuration,i,j,k,l):
    ## This gives the configurations which are connected to the ref_configuration(ket)
    ### via the excitation operator e(ij,kl) (chemists notation) 
    drt = np.array(drt, dtype=int) 
    # Get the tail and head node positions in the DRT  
    tail = min(i,j,k,l) 
    head = max(i,j,k,l) 
    start = np.zeros(3,dtype=int) 
    end = np.zeros(3,dtype=int) 
    # p, q start at 0, -1 for indexing step vector
    for a in range(tail-1): 
        start += step_vectors[ ref_configuration[a] ] 
    for a in range(head): 
        end += step_vectors[ ref_configuration[a] ] 
    start_ind = int(np.where((drt[:,:3] == start ).all(axis=1))[0][0]) 
    end_ind = int(np.where((drt[:,:3] == end ).all(axis=1))[0][0])
    nmo = np.sum(drt[0,:3]) 

    seta = set(range(min(i,j), max(i,j)+1)) 
    setb = set(range(min(k,l), max(k,l)+1))
    S1 = seta & setb 
    S2 = seta.union(setb) - S1 
    ## I want this in terms of levels - these are in terms of levels!
    # how then do i get a 2D array in which  
 
    connected_configs = []   
    ## Catching the diagonal cases  
    if (i==j) or (k==l): 
        if (i==j) and (k==l): 
            connected_configs = [ref_configuration] 
        elif (i==j): 
            connected_configs = one_body_selected_csf_basis(drt, ref_configuration,k,l) 
        elif (k==l): 
            connected_configs = one_body_selected_csf_basis(drt, ref_configuration,i,j) 
        if j==k:
            connected_configs += one_body_selected_csf_basis(drt, ref_configuration,i,l)
        return connected_configs  

    #print("i,j", i,j)
    #print("k,l", k,l)
    #print("S1", S1)
    #print("S2", S2)

    ## So here we iterate through the DRT levels, and then evalaute which are allowed paths
    ## 
    spawn = [ [],int(start_ind+1) ]  
    configurations = [ spawn ]  
    #for level in range(drt[start_ind,3], drt[end_ind,3]):
    ## we dont need the last head value!  
    for level in sorted(S1 | S2 )[:-1]: 
        ref_step = ref_configuration[level-1]
        new_configurations = [] 
        lexical_inds = [] 
        for conf in configurations:
            if conf[1]-1 not in lexical_inds:  
                lexical_inds.append(conf[1]-1) 
        lexical_inds  = np.array(lexical_inds) 
        
        ## Identify allowed steps  
        ## this is the key difference between the loops in the one body and two body steps 
        if level in S1:
            allowed_steps =  two_body_drt_step(ref_step,level,i,j,k,l) 
        elif level in S2: 
            if level in seta:  
                allowed_steps = one_body_drt_step(ref_step, level, i,j)  
            elif level in setb:  
                allowed_steps = one_body_drt_step(ref_step, level, i,j)  
        ### 
        for index in lexical_inds:
            if level == head - 1:
                steps = np.argwhere(drt[index,4:]==end_ind+1).reshape(-1)
            else: 
                steps = np.argwhere(drt[index,4:]!=0).reshape(-1)
         
            steps = [ x for x in steps if x in allowed_steps] 
            
            for configuration in configurations: 
                if configuration[1]==index+1: 
                    if len(steps) == 0 :
                        # Remove this from the configurations list! 
                        configurations = [ x for x in configurations if x[1]!=index+1 ] 
                    else: 
                        for step in steps: 
                            new_configurations.append( [configuration[0]+[step], drt[index,4+step] ] )
        configurations = new_configurations 
        print(configurations)  
    
    paths = np.zeros((len(configurations),nmo), dtype=int)
    configs = [] 
    for ind, path in enumerate(configurations): 
        if path[1]==end_ind+1:
            paths[ind,:tail-1] = ref_configuration[:tail-1]
            paths[ind,head:] = ref_configuration[head:]
            paths[ind,tail-1:head] = np.array(path[0]) 
            configs.append(paths[ind,:].copy()) 
        else: 
            print(f"path {ind} did not converge" ) 
    #return paths
    print("configs", configs)  
    return configs 

### These are the most important functions 
def one_body_drt_step(ref_step,level,i,j): 
    # Indices = [ i,j ]  
    # Check if R or L operator 
    if i < j : 
        oclass = "R"
    elif i > j: 
        oclass = "L" 
    else: 
        raise ("Error: one_body_drt_step doesnt deal with diagonal values shouldve been caught earlier")  

    ## Check if loop head, tail or main body     
    if level == min(i,j): 
        oclass = "t" + oclass 
    elif level == max(i,j): 
        oclass = "h" + oclass 
   
    allowed_steps = [] 
    if oclass=="tR" or oclass=="hL":
        if ref_step==0: 
            allowed_steps = [1,2] 
        elif ref_step==1 or ref_step==2: 
            allowed_steps = [0] 
    elif oclass=="hR" or oclass=="tL":  
        if ref_step==3: 
            allowed_steps = [1,2] 
        elif ref_step==1 or ref_step==2: 
            allowed_steps = [0] 
    elif oclass=="R" or oclass=="L": 
        if ref_step==0 or ref_step==3: 
            allowed_steps = [ref_step] 
        elif ref_step==1 or ref_step==2: 
            allowed_steps = [1,2]
 
    return allowed_steps 

def two_body_drt_step(ref_step,level,i,j,k,l): 
    ## 
    if i - j > 0: 
        a_class = "L" 
    elif i - j < 0: 
        a_class = "R" 
    else: 
        raise ("Error: two_body_drt_step doesnt deal with diagonal values shouldve been caught earlier")  
    if k - l > 0: 
        b_class = "L" 
    elif k - l < 0: 
        b_class = "R" 
    else: 
        raise ("Error: two_body_drt_step doesnt deal with diagonal values shouldve been caught earlier")  
    op_classes = [ a_class, b_class ] 
    heads = [max(i,j), max(k,l)]
    tails = [min(i,j), min(k,l)]
    if level == heads[0]: 
        op_classes[0] = "h"+op_classes[0]      
    if level == heads[1]: 
        op_classes[1] = "h"+op_classes[1]      
    if level == tails[0]: 
        op_classes[0] = "t"+op_classes[0]      
    if level == tails[1]: 
        op_classes[1] = "t"+op_classes[1]   

    allowed_steps = [] 
    if ( (op_classes==["hR", "hR" ]) 
        or (op_classes==["tL","tL"]) 
        or (("hR" in op_classes) and ("tL" in op_classes)) 
        ):
        if ref_step == 3 : 
            allowed_steps = [0] 
    elif ( (op_classes==["tR", "tR" ]) 
            or (op_classes==["hL","hL"])
            or (("tR" in op_classes) and ("hL" in op_classes)) ):
        if ref_step == 0 : 
            allowed_steps = [3] 
    elif(  (("hR" in op_classes) and ("hL" in op_classes)) 
        or (("tR" in op_classes) and ("tL" in op_classes)) 
        or (("tR" in op_classes) and ("hR" in op_classes))
        or (("tL" in op_classes) and ("hL" in op_classes)) 
        or (op_classes==["R","R"]) 
        or (op_classes==["L","L"]) 
        or (("R" in op_classes) and ("L" in op_classes)) ):  
        if ref_step==0 or ref_step==3: 
            allowed_steps = [ref_step] 
        elif ref_step==1 or ref_step==2: 
            allowed_steps = [1,2]
    
    elif ( (("R" in op_classes) and ("hR" in op_classes)) 
           or (("L" in op_classes) and ("tL" in op_classes))
           or (("hR" in op_classes) and ("L" in op_classes)) 
           or (("R" in op_classes) and ("tL" in op_classes)) ):
        if ref_step==1 or ref_step==2: 
            allowed_steps=[0]
        elif ref_step==3: 
            allowed_steps=[1,2]

    elif ( (("L" in op_classes) and ("hL" in op_classes)) 
          or (("R" in op_classes) and ("tR" in op_classes)) 
          or (("R" in op_classes) and ("hL" in op_classes))
          or (("tR" in op_classes) and ("L" in op_classes)) ):
        if ref_step==1 or ref_step==2: 
            allowed_steps=[3]
        elif ref_step==0: 
            allowed_steps=[1,2]
    else: 
        print("Fell through Error", op_classes)
    
    return allowed_steps 
