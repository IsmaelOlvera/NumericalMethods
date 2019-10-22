# -*- coding: utf-8 -*-
## activity 07


import sys
from pathlib import Path

parentPath  = Path().absolute().parent
siblingPath = parentPath.joinpath('modules')
sys.path.append( str(siblingPath) )

import numpy as np
import time

from decompositionLU import applications



if __name__ == '__main__':
    
    matA     = np.array( [ [18.,22.,54.,42.], [22.,70.,86.,62], [54.,86.,174.,134], [42.,62.,134.,106] ] )
    
    start_D   = time.time()
    matAinv_D = applications.findInverse(matA,"doolittle")
    end_D     = time.time()
    elapsedTime_D = end_D - start_D
    
    start_C   = time.time()
    matAinv_C = applications.findInverse(matA,"cholesky")
    end_C     = time.time()
    elapsedTime_C = end_C - start_C
    
    print("Matrix A: \n", matA, "\n")
    
    print("Matrix A ^ {-1} Doolittle: \n", matAinv_D)
    print("\t Elapsed time ", elapsedTime_D, "seconds \n")
    
    print("Matrix A ^ {-1} Cholesky: \n" , matAinv_C)
    print("\t Elapsed time ", elapsedTime_C, "seconds \n")
    
    print("Cholesky's Method was ", elapsedTime_D / elapsedTime_C, " times faster than Dooolitte's one. \n")
