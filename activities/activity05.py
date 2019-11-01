# -*- coding: utf-8 -*-
## activity 05


import sys
from pathlib import Path

parentPath  = Path().absolute().parent
siblingPath = parentPath.joinpath('modules')
sys.path.append( str(siblingPath) )

import numpy as np

from basics              import verification
from gaussianElimination import simpleGE



if __name__ == '__main__':

    matA = np.array( [ [4.,1.,2.,-3.,5.], [-3.,3.,-1.,4.,-2.], [-1.,2.,5.,1.,3.], [5.,4.,3.,-1.,2.], [1.,-2.,3.,-4.,5.] ] )
    vecB = np.array( [ -16.,20.,-4.,-10.,3. ] )
        
    vecSol = simpleGE.solve(matA,vecB)
    
    print("Matrix A: \n", matA, "\n")
    print("Vector b: \n", vecB, "\n")
    print("Vector Sol: \n", vecSol, "\n")
    
    print( verification.verifySystemSolution(matA,vecB,vecSol) )
