import numpy as np

import matplotlib.pyplot as plt



inputFile = open("actividad13_datos.in","r")

x,y,sigma = [],[],[]

for line in inputFile.readlines():
    x_i,y_i,sigma_i = map(float,line.strip().split())
    x.append(x_i) 
    y.append(y_i)
    sigma.append(sigma_i)
    
S_0  = sum( [ 1.0  / sigma[i]**2 for i in range(len(sigma)) ] )
S_x  = sum( [ x[i] / sigma[i]**2 for i in range(len(sigma)) ] )
S_y  = sum( [ y[i] / sigma[i]**2 for i in range(len(sigma)) ] )
S_xx = sum( [  x[i]**2  / sigma[i]**2 for i in range(len(sigma)) ] )
S_xy = sum( [ x[i]*y[i] / sigma[i]**2 for i in range(len(sigma)) ] )

a = (S_0*S_xy - S_x*S_y) / (S_0*S_xx - S_x*S_x) 
b = (  S_y    - S_x* a ) / S_0

lineX = np.arange(-1.,11.,0.1)
lineY = a*lineX + b

plt.plot(lineX,lineY, color='blue')
plt.errorbar(x,y, yerr=sigma, fmt='none', color='green')
plt.plot(x,y, color='red', linestyle='', marker='o', markersize=3)
plt.show()
