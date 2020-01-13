import numpy as np

import matplotlib.pyplot as plt


# Calcula el polinomio interpolador de lagrange
def interpolacionLagrange(xn,yn):
    
    aux         = np.poly1d([0])
    polinomio   = np.poly1d([0])
    componentes = []
    
    for i in range( len(xn) ):
        
        numerador   = np.poly1d([1])
        denominador = 1.0
        
        for j in range(len(xn)):
            if( i != j ):  
                aux          = np.poly1d( [1,-1.*xn[j] ] )
                numerador   *= aux
                denominador *= (xn[i] - xn[j])
        
        nuevoComponente = (numerador/denominador) * yn[i]
        
        componentes.append(nuevoComponente)
        polinomio += nuevoComponente
        
    return componentes,polinomio



# Valores primer ejercicio
#xn = np.array( [-1.,2.,3.,5.] )
#yn = np.array( [ 0.,1.,1.,2.] )
#exampleX = np.arange(-1.5, 5.5, 0.1)

# Valores segundo ejercicio
xn = np.array( [1850.,1900.,2000.,2010.] )
yn = np.array( [ 283.,291.,370.,388.] )
exampleX = np.arange(1840, 2020, 0.1)


colores = [ 'g', 'b', 'y', 'c', 'k']

componentes,polinomio = interpolacionLagrange(xn,yn)


# Gráfica de los polinomios componentes

for i in range( len(componentes) ):
    exampleY = componentes[i]([exampleX])[0]
    plt.plot(exampleX,exampleY,colores[i])

plt.plot(xn,yn,'or')
plt.show()


# Gráfica del polinomio interpolador

exampleY = polinomio([exampleX])[0]

plt.plot(exampleX,exampleY)
plt.plot(xn,yn,'or')
plt.show()


# Continuación segundo ejercicio
print( "Concentración en 1950: " + str(polinomio(1950)) )
print( "Concentración en 2016: " + str(polinomio(2016)) )
