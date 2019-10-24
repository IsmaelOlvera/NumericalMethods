# -*- coding: utf-8 -*-
## module basics.norm


import numpy as np
import math


########## INPUT  ##########
## vecX  -> Vector [n]
########## OUTPUT ##########
## normX -> Integer  |  normX = sqrt( x_1^{2} + ... + x_n^{2} )
def euclidean(vecX):
    
    normX = math.sqrt( sum( [ vecX[k]**2 for k in range( len(vecX) ) ] ) )
    
    return normX



if __name__ == "__main__":
    
    vecX = np.array( [-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.] )
    
    normX = euclidean(vecX)
    
    print("Vector x: \n", vecX, "\n")
    print("Norm: \n",normX,"\n")
