import numpy as np 
def A(b,x,y): 
    return np.sqrt((b+x)/(b+y))

def C(b,x): 
    return np.sqrt((b+x-1)*(b+x+1))/(b+x)

# Table one encodes d',d, (W, hR, hL, tR, tL) 
table_one = np.full((4,4,5),lambda b:0, dtype=object)

#Set W elements 
table_one[0,0,0]=lambda b: 0
table_one[1,1,0]=lambda b: 1
table_one[2,2,0]=lambda b: 1
table_one[3,3,0]=lambda b: 2

#Set hR elements 
table_one[0,1,1]=lambda b: 1
table_one[0,2,1]=lambda b: 1
table_one[1,3,1]=lambda b: A(b,0,1)
table_one[2,3,1]=lambda b: A(b,2,1)

#Set hL elements 
table_one[1,0,2]=lambda b: 1
table_one[2,0,2]=lambda b: 1
table_one[3,1,2]=lambda b: A(b,0,1)
table_one[3,2,2]=lambda b: A(b,2,1)

#Set tR elements 
table_one[1,0,3]=lambda b: 1
table_one[2,0,3]=lambda b: 1
table_one[3,1,3]=lambda b: A(b,1,0)
table_one[3,2,3]=lambda b: A(b,1,2)

#Set tL elements 
table_one[0,1,4]=lambda b: 1
table_one[0,2,4]=lambda b: 1
table_one[1,3,4]=lambda b: A(b,2,1)
table_one[2,3,4]=lambda b: A(b,0,1)

# Table two encodes d',d, (R,l) ,(Db=-1, Db=1)
table_two = np.full((4,4,2,2),lambda b:0, dtype=object)

# Define Rs 
# Deltab = -1 
table_two[0,0,0,0] = lambda b: 1  
table_two[1,1,0,0] = lambda b: -1  
table_two[1,2,0,0] = lambda b: -1/(b+2)  
table_two[2,2,0,0] = lambda b: C(b,2)  
table_two[3,3,0,0] = lambda b: -1  

# Deltab = 1 
table_two[0,0,0,1] = lambda b: 1  
table_two[1,1,0,1] = lambda b: C(b,0)   
table_two[2,1,0,1] = lambda b: 1/b  
table_two[2,2,0,1] = lambda b: -1   
table_two[3,3,0,1] = lambda b: -1 

# Define Ls 
# Deltab = -1 
table_two[0,0,1,0] = lambda b: 1  
table_two[1,1,1,0] = lambda b: C(b,1)  
table_two[1,2,1,0] = lambda b: 1/(b+1)  
table_two[2,2,1,0] = lambda b: -1  
table_two[3,3,1,0] = lambda b: -1  

# Deltab = 1 
table_two[0,0,1,1] = lambda b: 1  
table_two[1,1,1,1] = lambda b: -1   
table_two[2,1,1,1] = lambda b: -1/(b+1)  
table_two[2,2,1,1] = lambda b: C(b,1)  
table_two[3,3,1,1] = lambda b: -1 
