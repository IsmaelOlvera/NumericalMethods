# -*- coding: utf-8 -*-
## activity 06.2



import sys
from pathlib import Path

parentPath  = Path().absolute().parent
siblingPath = parentPath.joinpath('modules')
sys.path.append( str(siblingPath) )

import numpy as np

import decompositionLU
import verification



if __name__ == '__main__':
    
    matA    = np.array( [ [1.,-3.,7.], [0.,3.,-2.], [-2.,6.,1.] ] )
    matAinv = decompositionLU.findInverse(matA)
    
    print("Matrix A: \n", matA, "\n")
    print("Matrix A ^ {-1}: \n", matAinv, "\n")
    
    print( verification.verifyInverse(matA,matAinv) )
