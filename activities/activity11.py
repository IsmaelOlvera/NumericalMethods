import math
import numpy as np

import matplotlib.pyplot as plt


def diferenciasDividias(x,y):
    
    n     = len(x)
    tabla = np.zeros((n,n))
    
    for i in range(n):
        tabla[i,0] = y[i]
        
    for j in range(1,n):
        for i in range(n-j):
            numerador   = tabla[i+1,j-1] - tabla[i,j-1]
            denominador = x[i+j] - x[i]
            tabla[i,j]  = numerador / denominador
    
    return tabla[0,:]


def interpolacionNetwon(c,x):
    
    n = len(x)
    
    polinomioSecundario   = np.poly1d(  [1.] )
    polinomioInterpolador = np.poly1d( [c[0]] )
    
    for i in range(n-1):
        polinomioSecundario   *= np.poly1d( [1.,-1.*x[i]] )
        polinomioInterpolador += primeraFila[i+1]*polinomioSecundario
        
    return polinomioInterpolador


# Valores ejemplo clase
#x = [ -1.,0.,2.,3. ]
#y = [ -8.,3.,1.,12. ]

# Valores ejemplo de prueba
#x = [ 1.,2.,3.,4. ]
#y = [ 6.,9.,2.,5. ]

# Valores de Actividad 10 [Inciso f]
x_puntos = [ 0.6,0.7,0.8,0.9,1.0 ]
y_puntos = [ math.e**x_puntos[i] for i in range(len(x_puntos)) ]


# Cálculo de Polinomio Interpolador [Inciso a]
primeraFila           = diferenciasDividias(x_puntos,y_puntos)
polinomioInterpolador = interpolacionNetwon(primeraFila,x_puntos)

print("Polinomio Interpolador: \n" +  str(polinomioInterpolador) )


# Evaluación de puntos en [x_puntos_0,x_puntos_n] ; f(x) = e**x ; [Inciso b]
x_funcion = np.linspace(x_puntos[0],x_puntos[len(x_puntos)-1],101)
y_funcion = [ math.e**x_funcion[i] for i in range(len(x_funcion)) ]


# Evaluación de puntos en [x_puntos_0,x_puntos_n] ; g(x) = polinomioInterpolador ; [Inciso c]
x_polinomio = x_funcion
y_polinomio = polinomioInterpolador(x_polinomio)

# Evaluación de puntos en [x_puntos_0,x_puntos_n] ; h(x) = f(x) - g(x) ; [Inciso c]
x_delta = x_funcion
y_delta = x_funcion - x_polinomio


# Gráfica de la función y Polinomio Interpolador en [x_puntos_0,x_puntos_n] y además puntos originales [Inciso d]
plt.plot(x_funcion,y_funcion, color='blue')
plt.plot(x_polinomio,y_polinomio, color='orange', linestyle='dotted')
plt.plot(x_puntos,y_puntos,'ro')
plt.show()

# Gráfica de h(x) = f(x) - g(x) en [x_puntos_0,x_puntos_n] [Inciso e]
x_delta = x_funcion
y_delta = x_funcion - x_polinomio

plt.plot(x_delta,y_delta, color='blue')
plt.show()


# Calcular valor en 0.82 y 0.98 con Polinomio Interpolador [Inciso g]
print( "El valor estimado por el Polinomio Interpolador en 0.82 es:" + str(polinomioInterpolador(0.82)) )
print( "El valor estimado por el Polinomio Interpolador en 0.98 es:" + str(polinomioInterpolador(0.98)) )


# Cálculo de delta en dos intervalos diferentes [Inciso h]
x_delta2 = np.linspace(0.5,1.0,101)
y_delta2 = [ math.e**x_delta2[i] for i in range(len(x_delta2)) ] - polinomioInterpolador(x_delta2)

plt.plot(x_delta2,y_delta2)
plt.show()

x_delta3 = np.linspace(0.0,2.0,201)
y_delta3 = [ math.e**x_delta3[i] for i in range(len(x_delta3)) ] - polinomioInterpolador(x_delta3)

plt.plot(x_delta3,y_delta3)
plt.show()
