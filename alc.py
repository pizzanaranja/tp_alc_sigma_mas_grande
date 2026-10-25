
import numpy as np
def multiplicarMatrices(A,B):
    res = np.zeros((len(A),len(B[0])))
    for f in range(len(A)):
        for c in range(len(B[0])):
            sum = 0
            for fb in range(len(B)):
                sum += A[f][fb]*B[fb][c]
            res[f][c] = sum

    return res

def multiplicarAx(A,x):
    Ax = np.zeros(len(x))
    for i in range(len(A)):
        sum = 0
        for j in range(len(x)):
            sum += A[i][j]*x[j]
        Ax[i] = sum
    return Ax

def traspuesta(A):
    f,c = A.shape
    T = np.zeros((c,f))
    i = 0
    while i < c:
        T[i,:] = A[:,i]
        i+=1
    return T

def esSimetrica(A,error):
    f,c = A.shape
    if f == c:
        T = traspuesta(A)
        simetrica = True
        i = 0
        while simetrica and i < f:
            for j in range(c):
                simetrica = np.isclose(A[i,j], T[i,j],atol=error)
            i += 1
        
        return simetrica

def norma(x,p):
    norm = 0
    if p == "inf":
        for elem in x:
            if abs(elem) > norm:
                norm = abs(elem)
    else:
        suma = 0
        for elem in x:
            suma += abs(elem)**p
            norm = suma**(1/p)
    return norm

def producto_interno(x, y):
    if len(x) != len(y):
        return None
    prod = 0
    for i in range(len(x)):
        prod += x[i]*y[i]
    return prod

#### EJERCICIOS
def f_A(A, v, k):
    w = v.copy()
    for i in range(k):
        w = multiplicarAx(A,w)
        if norma(w, 2) > 0:
            w = w/norma(w, 2)
    return w

def metpot2k(A,tol=1e-15,K=1000):
    v = np.random.randn(len(A))
    v_m = f_A(A,v,2)
    e = producto_interno(v_m, v)
    k = 0
    while abs(e - 1)> tol and k < K:
        v = v_m
        v_m = f_A(A,v,2)
        e = producto_interno(v_m,v)
        k += 1
    Av_m = multiplicarAx(A, v_m)
    lamda = producto_interno(v_m, Av_m)
    error = e - 1
    return v_m, lamda, k

def diagRH(A,tol=1e-15,K=1e5):
    if not esSimetrica(A, tol):
        return None
    v, lamda, error = metpot2k(A, tol, K)
    if lamda<tol:
        return np.zeros((len(A),len(A))), np.zeros((len(A),len(A)))
    I = np.eye(len(A))
    u = I[0].copy()
    u -= v
    H = I - (2/norma(u,2)**2)*multiplicarMatrices(u.reshape((len(u),1)), traspuesta(u.reshape((len(u),1))))
    if len(A) == 2:
        S = H
        D = multiplicarMatrices(H, multiplicarMatrices(A, traspuesta(H)))
    else:
        B = multiplicarMatrices(H, multiplicarMatrices(A, traspuesta(H)))
        A_mon = B[1:, 1:]
        S_mon, D_mon = diagRH(A_mon, tol, K)
        D = np.eye(len(A))
        D[0][0] = lamda
        D[1:, 1:] = D_mon
        S1 = np.eye(len(A))
        S1[1:, 1:] = S_mon
        S = multiplicarMatrices(H, S1)    
    return S, D