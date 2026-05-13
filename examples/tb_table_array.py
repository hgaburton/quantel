import numpy as np 
from ob_table_array import A, C
t = np.sqrt(1/2) 
def B(b,p,q): 
    return np.sqrt( 2/((b+p)*(b+q)) )

def D(b,p): 
    return np.sqrt( ( (b+p-1)*(2*p +2) )/( (b+p)*(b+p+1) ) )

# Table one is for d',d, ({hRhR =tLtL},{hLhL==tRtR}, hRhL, tRtL ),(x=0, x=1)  
table_one = np.full((4,4,4,2),lambda b:0.0, dtype=object)
# hRhR
table_one[0,3,0,0]=lambda b: np.sqrt(2)
table_one[0,3,0,1]=lambda b: 0 

# hLhL
table_one[3,0,1,0]=lambda b: np.sqrt(2)
table_one[3,0,1,1]=lambda b: 0 

# hRhL, x=0
table_one[0,0,2,0]=lambda b: 0  
table_one[1,1,2,0]=lambda b: t
table_one[1,2,2,0]=lambda b: 0
table_one[2,1,2,0]=lambda b: 0
table_one[2,2,2,0]=lambda b: t
table_one[3,3,2,0]=lambda b: 2*t
# hRhL, x=1
table_one[0,0,2,1]=lambda b: 0  
table_one[1,1,2,1]=lambda b: -t*A(b,-1,1)
table_one[1,2,2,1]=lambda b: 1
table_one[2,1,2,1]=lambda b: 1
table_one[2,2,2,1]=lambda b: t*A(b,3,1)
table_one[3,3,2,1]=lambda b: 0

# tRtL, x=0
table_one[0,0,3,0]=lambda b: 0  
table_one[1,1,3,0]=lambda b: t
table_one[1,2,3,0]=lambda b: 0
table_one[2,1,3,0]=lambda b: 0
table_one[2,2,3,0]=lambda b: -t
table_one[3,3,3,0]=lambda b: -2*t
# tRtL, x=1
table_one[0,0,3,1]=lambda b: 0  
table_one[1,1,3,1]=lambda b: t*A(b,2,0)
table_one[1,2,3,1]=lambda b: C(b,2)
table_one[2,1,3,1]=lambda b: C(b,0)
table_one[2,2,3,1]=lambda b: -t*A(b,0,2)
table_one[3,3,3,1]=lambda b: 0

# recall for mixed contributions the relative sign of the contributions is invariant with respect to the order 
# for example: hRtL == tLhR ! 

# Table two is for  d',d, ({hRtR = tRhR}, {hLtL = tLhL}, hRtL, tRhL) ,(deltab=-1, deltab=1)  
table_two = np.full((4,4,4,2),lambda b:0, dtype=object)
# R operators 
table_two[1,1,0,0]=lambda b: 1
table_two[1,2,0,0]=lambda b: 1
table_two[2,1,0,1]=lambda b: 1
table_two[2,2,0,1]=lambda b: 1
table_two[3,3,0,0]=lambda b: 1
table_two[3,3,0,1]=lambda b: 1

# L operators 
table_two[1,1,1,1]=lambda b: 1
table_two[1,2,1,0]=lambda b: 1
table_two[2,1,1,1]=lambda b: 1
table_two[2,2,1,0]=lambda b: 1
table_two[3,3,1,0]=lambda b: 1
table_two[3,3,1,1]=lambda b: 1

# hRtL operators 
table_two[0,3,2,0]=lambda b: A(b,2,1)
table_two[0,3,2,1]=lambda b: A(b,0,1)

# tRhL operators 
table_two[0,3,3,0]=lambda b: A(b,1,1)
table_two[0,3,3,1]=lambda b: A(b,1,0)


###################################
##################################
# Table three is for d', d, (RhR,hRR, hLL,LhL, hRL, RhL) ,(deltab=-1, deltab=1), (x=0, x=1)   
table_three = np.full((4,4,6,2,2),lambda b:0, dtype=object)
# RhR , deltab = -1, x = 0 
table_three[0,1,0,0,0] = lambda b: 0
table_three[0,2,0,0,0] = lambda b: t*A(b,1,2)
table_three[1,3,0,0,0] = lambda b: -t
table_three[2,3,0,0,0] = lambda b: 0
# RhR , deltab = -1, x = 1
table_three[0,1,0,0,1] = lambda b: 1
table_three[0,2,0,0,1] = lambda b: t*A(b,3,2)
table_three[1,3,0,0,1] = lambda b: -t*A(b,0,2)
table_three[2,3,0,0,1] = lambda b: A(b,3,2)
# hRR , deltab = -1, x = 0 
table_three[0,1,1,0,0] = lambda b: 0
table_three[0,2,1,0,0] = lambda b: t*A(b,1,2)
table_three[1,3,1,0,0] = lambda b: -t
table_three[2,3,1,0,0] = lambda b: 0
# hRR , deltab = -1, x = 1
table_three[0,1,1,0,1] = lambda b: -1
table_three[0,2,1,0,1] = lambda b: -t*A(b,3,2)
table_three[1,3,1,0,1] = lambda b: t*A(b,0,2)
table_three[2,3,1,0,1] = lambda b: -A(b,3,2)


# RhR , deltab = 1, x = 0 
table_three[0,1,0,1,0] = lambda b: t*A(b,1,0)
table_three[0,2,0,1,0] = lambda b: 0
table_three[1,3,0,1,0] = lambda b: 0
table_three[2,3,0,1,0] = lambda b: -t
# RhR , deltab = 1, x = 1
table_three[0,1,0,1,1] = lambda b: -t*A(b,-1,0)
table_three[0,2,0,1,1] = lambda b: 1
table_three[1,3,0,1,1] = lambda b: A(b,-1,0)
table_three[2,3,0,1,1] = lambda b: t*A(b,2,0)
# hRR , deltab = 1, x = 0 
table_three[0,1,1,1,0] = lambda b: t*A(b,1,0)
table_three[0,2,1,1,0] = lambda b: 0
table_three[1,3,1,1,0] = lambda b: 0
table_three[2,3,1,1,0] = lambda b: -t
# hRR , deltab = 1, x = 1
table_three[0,1,1,1,1] = lambda b: t*A(b,-1,0)
table_three[0,2,1,1,1] = lambda b: -1
table_three[1,3,1,1,1] = lambda b: -A(b,-1,0)
table_three[2,3,1,1,1] = lambda b: -t*A(b,2,0)


# hLL , deltab = -1, x = 0 
table_three[1,0,2,0,0] = lambda b:t*A(b,2,1) 
table_three[2,0,2,0,0] = lambda b: 0
table_three[3,1,2,0,0] = lambda b: 0
table_three[3,2,2,0,0] = lambda b: -t
# LhL , deltab = -1, x = 0 
table_three[1,0,3,0,0] = lambda b:t*A(b,2,1) 
table_three[2,0,3,0,0] = lambda b: 0
table_three[3,1,3,0,0] = lambda b: 0
table_three[3,2,3,0,0] = lambda b: -t

# hLL , deltab = -1, x = 1 
table_three[1,0,2,0,1] = lambda b:-t*A(b,0,1) 
table_three[2,0,2,0,1] = lambda b: 1
table_three[3,1,2,0,1] = lambda b: A(b,0,1)
table_three[3,2,2,0,1] = lambda b: t*A(b,3,1)
# LhL , deltab = -1, x = 1 
table_three[1,0,3,0,1] = lambda b: t*A(b,0,1) 
table_three[2,0,3,0,1] = lambda b: -1
table_three[3,1,3,0,1] = lambda b: -A(b,0,1)
table_three[3,2,3,0,1] = lambda b: -t*A(b,3,1)


# hLL , deltab = 1, x = 0 
table_three[1,0,2,1,0] = lambda b: 0  
table_three[2,0,2,1,0] = lambda b: t*A(b,0,1)
table_three[3,1,2,1,0] = lambda b: -t
table_three[3,2,2,1,0] = lambda b: 0
# LhL , deltab = 1, x = 0 
table_three[1,0,3,1,0] = lambda b: 0  
table_three[2,0,3,1,0] = lambda b: t*A(b,0,1)
table_three[3,1,3,1,0] = lambda b: -t
table_three[3,2,3,1,0] = lambda b: 0


# hLL , deltab = 1, x = 1 
table_three[1,0,2,1,1] = lambda b: 1  
table_three[2,0,2,1,1] = lambda b: t*A(b,2,1)
table_three[3,1,2,1,1] = lambda b: -t*A(b,-1,1)
table_three[3,2,2,1,1] = lambda b: A(b,2,1)
# LhL , deltab = 1, x = 1 
table_three[1,0,3,1,1] = lambda b: -1  
table_three[2,0,3,1,1] = lambda b: -t*A(b,2,1)
table_three[3,1,3,1,1] = lambda b: t*A(b,-1,1)
table_three[3,2,3,1,1] = lambda b: -A(b,2,1)



# hRL , deltab = -1, x = 0 
table_three[0,1,4,0,0] = lambda b: 0 
table_three[0,2,4,0,0] = lambda b: t
table_three[1,3,4,0,0] = lambda b: t*A(b,2,1)
table_three[2,3,4,0,0] = lambda b: 0
# hRL , deltab = -1, x = 1 
table_three[0,1,4,0,0] = lambda b: 1
table_three[0,2,4,0,0] = lambda b: t*A(b,3,1)
table_three[1,3,4,0,0] = lambda b: t*A(b,0,1)
table_three[2,3,4,0,0] = lambda b: -A(b,2,1)


# hRL , deltab = 1, x = 0 
table_three[0,1,4,1,0] = lambda b: t 
table_three[0,2,4,1,0] = lambda b: 0
table_three[1,3,4,1,0] = lambda b: 0
table_three[2,3,4,1,0] = lambda b: t*A(b,0,1)
# hRL , deltab = -1, x = 1 
table_three[0,1,4,1,1] = lambda b: -t*A(b,-1,1)
table_three[0,2,4,1,1] = lambda b: 1
table_three[1,3,4,1,1] = lambda b: -A(b,0,1)
table_three[2,3,4,1,1] = lambda b: -t*A(b,2,1)


# RhL , deltab = -1, x = 0 
table_three[1,0,5,0,0] = lambda b: t 
table_three[2,0,5,0,0] = lambda b: 0
table_three[3,1,5,0,0] = lambda b: 0
table_three[3,2,5,0,0] = lambda b: t*A(b,1,2)
# RhL , deltab = -1, x = 1 
table_three[1,0,5,0,1] = lambda b: -t*A(b,0,2) 
table_three[2,0,5,0,1] = lambda b: 1
table_three[3,1,5,0,1] = lambda b: -A(b,1,2)
table_three[3,2,5,0,1] = lambda b: -t*A(b,3,2)

# RhL , deltab = 1, x = 0 
table_three[1,0,5,1,0] = lambda b: 0 
table_three[2,0,5,1,0] = lambda b: t
table_three[3,1,5,1,0] = lambda b: t*A(b,1,0)
table_three[3,2,5,1,0] = lambda b: 0
# RhL , deltab = 1, x = 1 
table_three[1,0,5,1,1] = lambda b: 1 
table_three[2,0,5,1,1] = lambda b: t*A(b,2,0)
table_three[3,1,5,1,1] = lambda b: t*A(b,-1,0)
table_three[3,2,5,1,1] = lambda b: -A(b,1,0)


###########################################
###########################################


# Table four is for d',d, (tRR, RtR, LtL,tLL,RR,LL,RtL, tRL, RL) ,(deltab=-2, deltab=0, deltab=2), (x=0, x=1)   
table_four = np.full((4,4,9,3,2), lambda b: 0, dtype=object)
# tRR , deltab = -2, x = 1 
table_four[1,0,0,0,1] = lambda b: 1
table_four[2,0,0,0,1] = lambda b: 0 
table_four[3,1,0,0,1] = lambda b: 0
table_four[3,2,0,0,1] = lambda b: A(b,1,2)
# RtR , deltab = -2, x = 1 
table_four[1,0,1,0,1] = lambda b: 1
table_four[2,0,1,0,1] = lambda b: 0 
table_four[3,1,1,0,1] = lambda b: 0
table_four[3,2,1,0,1] = lambda b: -A(b,1,2)

# tRR , deltab = 0, x = 0 
table_four[0,1,0,1,0] = lambda b: t*A(b,0,1)
table_four[0,2,0,1,0] = lambda b: t*A(b,2,1)
table_four[1,3,0,1,0] = lambda b: -t
table_four[2,3,0,1,0] = lambda b: -t
# RtR , deltab = 0, x = 0 
table_four[1,0,1,1,0] = lambda b: t*A(b,0,1)
table_four[2,0,1,1,0] = lambda b: t*A(b,2,1)
table_four[3,1,1,1,0] = lambda b: -t
table_four[3,2,1,1,0] = lambda b: -t


# tRR , deltab = 0, x = 1 
table_four[1,0,0,1,1] = lambda b: t*A(b,2,1)
table_four[2,0,0,1,1] = lambda b: -t*A(b,0,1)
table_four[3,1,0,1,1] = lambda b: t*A(b,2,0)
table_four[3,2,0,1,1] = lambda b: -t*A(b,0,2)
# RtR , deltab = 0, x = 1 
table_four[1,0,1,1,1] = lambda b: -t*A(b,2,1)
table_four[2,0,1,1,1] = lambda b: t*A(b,0,1)
table_four[3,1,1,1,1] = lambda b: -t*A(b,2,0)
table_four[3,2,1,1,1] = lambda b: t*A(b,0,2)

# tRR , deltab = 2, x = 1 
table_four[1,0,0,2,1] = lambda b: 0 
table_four[2,0,0,2,1] = lambda b: 1
table_four[3,1,0,2,1] = lambda b: A(b,1,0)
table_four[3,2,0,2,1] = lambda b: 0
# RtR , deltab = 2, x = 1 
table_four[1,0,1,2,1] = lambda b: 0 
table_four[2,0,1,2,1] = lambda b: -1
table_four[3,1,1,2,1] = lambda b: -A(b,1,0)
table_four[3,2,1,2,1] = lambda b: 0
####################
####################
# LtL , deltab = -2, x = 1 
table_four[0,1,2,0,1] = lambda b: 0
table_four[0,2,2,0,1] = lambda b: 1 
table_four[1,3,2,0,1] = lambda b: A(b,3,2)
table_four[2,3,2,0,1] = lambda b: 0
# tLL , deltab = -2, x = 1 
table_four[0,1,3,0,1] = lambda b: 0
table_four[0,2,3,0,1] = lambda b: -1 
table_four[1,3,3,0,1] = lambda b: -A(b,3,2)
table_four[2,3,3,0,1] = lambda b: 0


# LtL , deltab = 0, x = 0 
table_four[0,1,2,1,0] = lambda b: t*A(b,0,1)
table_four[0,2,2,1,0] = lambda b: t*A(b,2,1)
table_four[1,3,2,1,0] = lambda b: -t
table_four[2,3,2,1,0] = lambda b: -t
# tLL , deltab = 0, x = 0 
table_four[0,1,3,1,0] = lambda b: t*A(b,0,1)
table_four[0,2,3,1,0] = lambda b: t*A(b,2,1)
table_four[1,3,3,1,0] = lambda b: -t
table_four[2,3,3,1,0] = lambda b: -t


# LtL , deltab = 0, x = 1 
table_four[0,1,2,1,1] = lambda b: t*A(b,2,1)
table_four[0,2,2,1,1] = lambda b: -t*A(b,0,1)
table_four[1,3,2,1,1] = lambda b: t*A(b,2,0)
table_four[2,3,2,1,1] = lambda b: -t*A(b,0,2)
# tLL , deltab = 0, x = 1 
table_four[0,1,3,1,1] = lambda b: -t*A(b,2,1)
table_four[0,2,3,1,1] = lambda b: t*A(b,0,1)
table_four[1,3,3,1,1] = lambda b: -t*A(b,2,0)
table_four[2,3,3,1,1] = lambda b: t*A(b,0,2)


# LtL , deltab = -2, x = 1 
table_four[0,1,2,2,1] = lambda b: 1
table_four[0,2,2,2,1] = lambda b: 0 
table_four[1,3,2,2,1] = lambda b: 0
table_four[2,3,2,2,1] = lambda b: A(b,-1,0)
# tLL , deltab = -2, x = 1 
table_four[0,1,3,2,1] = lambda b: -1
table_four[0,2,3,2,1] = lambda b: 0 
table_four[1,3,3,2,1] = lambda b: 0
table_four[2,3,3,2,1] = lambda b: -A(b,-1,0)
########################
########################

# RR , deltab = -2, x = 1 
table_four[0,0,4,0,1] = lambda b: 1 
table_four[1,1,4,0,1] = lambda b: 1
table_four[1,2,4,0,1] = lambda b: B(b,2,3)
table_four[2,1,4,0,1] = lambda b: 0
table_four[2,2,4,0,1] = lambda b: D(b,2)
table_four[3,3,4,0,1] = lambda b: 1

# RR , deltab = 0, x = 0 
table_four[0,0,4,1,0] = lambda b: 1 
table_four[1,1,4,1,0] = lambda b: -1
table_four[1,2,4,1,0] = lambda b: 0
table_four[2,1,4,1,0] = lambda b: 0
table_four[2,2,4,1,0] = lambda b: -1
table_four[3,3,4,1,0] = lambda b: 1
# RR , deltab = 0, x = 1 
table_four[0,0,4,1,1] = lambda b: 1 
table_four[1,1,4,1,1] = lambda b: -D(b,0)
table_four[1,2,4,1,1] = lambda b: B(b,1,2)
table_four[2,1,4,1,1] = lambda b: B(b,0,1)
table_four[2,2,4,1,1] = lambda b: -D(b,1)
table_four[3,3,4,1,1] = lambda b: 1

# RR , deltab = 2, x = 1 
table_four[0,0,4,2,1] = lambda b: 1 
table_four[1,1,4,2,1] = lambda b: D(b,-1)
table_four[1,2,4,2,1] = lambda b: 0
table_four[2,1,4,2,1] = lambda b: B(b,-1,0)
table_four[2,2,4,2,1] = lambda b: 1
table_four[3,3,4,2,1] = lambda b: 1

########################
########################

# LL , deltab = -2, x = 1 
table_four[0,0,5,0,1] = lambda b: 1 
table_four[1,1,5,0,1] = lambda b: D(b,1)
table_four[1,2,5,0,1] = lambda b: B(b,1,2)
table_four[2,1,5,0,1] = lambda b: 0
table_four[2,2,5,0,1] = lambda b: 1
table_four[3,3,5,0,1] = lambda b: 1


# LL , deltab = 0, x = 0 
table_four[0,0,5,1,0] = lambda b: 1 
table_four[1,1,5,1,0] = lambda b: -1 
table_four[1,2,5,1,0] = lambda b: 0
table_four[2,1,5,1,0] = lambda b: 0
table_four[2,2,5,1,0] = lambda b: -1
table_four[3,3,5,1,0] = lambda b: 1
# LL , deltab = 0, x = 1 
table_four[0,0,5,1,1] = lambda b: 1 
table_four[1,1,5,1,1] = lambda b: -D(b,0) 
table_four[1,2,5,1,1] = lambda b: B(b,0,1)
table_four[2,1,5,1,1] = lambda b: B(b,1,2)
table_four[2,2,5,1,1] = lambda b: -D(b,1)
table_four[3,3,5,1,1] = lambda b: 1


# LL , deltab = 2, x = 1 
table_four[0,0,5,2,1] = lambda b: 1 
table_four[1,1,5,2,1] = lambda b: 1 
table_four[1,2,5,2,1] = lambda b: 0
table_four[2,1,5,2,1] = lambda b: B(b,0,1)
table_four[2,2,5,2,1] = lambda b: D(b,0)
table_four[3,3,5,2,1] = lambda b: 1


#######################################
######################################

# RtL , deltab = -2, x = 1 
table_four[0,1,6,0,1] = lambda b: 0
table_four[0,2,6,0,1] = lambda b: C(b,2) 
table_four[1,3,6,0,1] = lambda b: -A(b,3,2)
table_four[2,3,6,0,1] = lambda b: 0 

# RtL , deltab = 0, x = 0 
table_four[0,1,6,1,0] = lambda b: -t
table_four[0,2,6,1,0] = lambda b: -t 
table_four[1,3,6,1,0] = lambda b: -t*A(b,0,1)
table_four[2,3,6,1,0] = lambda b: -t*A(b,2,1)
# RtL , deltab = 0, x = 0 
table_four[0,1,6,1,1] = lambda b: t*A(b,2,0)
table_four[0,2,6,1,1] = lambda b: -t*A(b,0,2)
table_four[1,3,6,1,1] = lambda b: -t*A(b,2,1)
table_four[2,3,6,1,1] = lambda b: t*A(b,0,1)

# RtL , deltab = 2, x = 1 
table_four[0,1,6,2,1] = lambda b: C(b,0)
table_four[0,2,6,2,1] = lambda b: 0 
table_four[1,3,6,2,1] = lambda b: 0
table_four[2,3,6,2,1] = lambda b: -A(b,-1,0) 


############
# tRL , deltab = -2, x = 1 
table_four[1,0,7,0,1] = lambda b: C(b,2)
table_four[2,0,7,0,1] = lambda b: 0 
table_four[3,1,7,0,1] = lambda b: 0
table_four[3,2,7,0,1] = lambda b: -A(b,1,2) 

# tRL , deltab = 0, x = 0 
table_four[1,0,7,1,0] = lambda b: -t 
table_four[2,0,7,1,0] = lambda b: -t
table_four[3,1,7,1,0] = lambda b: -t*A(b,0,1)
table_four[3,2,7,1,0] = lambda b: -t*A(b,2,1)
# tRL , deltab = 0, x = 1 
table_four[1,0,7,1,1] = lambda b: t*A(b,2,0) 
table_four[2,0,7,1,1] = lambda b: -t*A(b,0,2)
table_four[3,1,7,1,1] = lambda b: -t*A(b,2,1)
table_four[3,2,7,1,1] = lambda b: t*A(b,0,1)

# tRL , deltab = 2, x = 1 
table_four[1,0,7,2,1] = lambda b: 0
table_four[2,0,7,2,1] = lambda b: C(b,0) 
table_four[3,1,7,2,1] = lambda b: -A(b,1,0)
table_four[3,2,7,2,1] = lambda b: 0 



###############################
# RL , deltab = -2, x = 1 
table_four[0,0,8,0,1] = lambda b: 1 
table_four[1,1,8,0,1] = lambda b: -C(b,2)
table_four[1,2,8,0,1] = lambda b: - np.sqrt(2) / (b+2) 
table_four[2,1,8,0,1] = lambda b: 0
table_four[2,2,8,0,1] = lambda b: -C(b,2)
table_four[3,3,8,0,1] = lambda b: 1 


# RL , deltab = 0, x = 0 
table_four[0,0,8,1,0] = lambda b: 1 
table_four[1,1,8,1,0] = lambda b: 1
table_four[1,2,8,1,0] = lambda b: 0
table_four[2,1,8,1,0] = lambda b: 0
table_four[2,2,8,1,0] = lambda b: 1
table_four[3,3,8,1,0] = lambda b: 1
# RL , deltab = 0, x = 1 
table_four[0,0,8,1,1] = lambda b: 1 
table_four[1,1,8,1,1] = lambda b: D(b,0)
table_four[1,2,8,1,1] = lambda b: -B(b,0,2)
table_four[2,1,8,1,1] = lambda b: -B(b,0,2)
table_four[2,2,8,1,1] = lambda b: D(b,1)
table_four[3,3,8,1,1] = lambda b: 1

# RL , deltab = 2, x = 1 
table_four[0,0,8,2,1] = lambda b: 1 
table_four[1,1,8,2,1] = lambda b: -C(b,0)
table_four[1,2,8,2,1] = lambda b: 0 
table_four[2,1,8,2,1] = lambda b: -np.sqrt(2)/b 
table_four[2,2,8,2,1] = lambda b: -C(b,0)
table_four[3,3,8,2,1] = lambda b: 1 
