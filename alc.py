
import numpy as np
#agrego los modulos
def error(x,y):
    return abs(np.float64(x) - np.float64(y))
'''Recibe dos numeros x e y, y calcula el error de aproximar x usando y en float64'''

def error_relativo(x,y):
    return abs(error(x,y)) / abs(np.float64(x))
'''Recibe dos numeros x e y, y calcula el error relativo de aproximar x usando y en float64'''
    
def matricesIguales(A, B):
    iguales = True
    f,c = A.shape
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        iguales = False
    else:
        i = 0
        while iguales and i < f:
            for j in range(c):
                iguales = iguales and np.isclose(A[i][j], B[i][j])
            i += 1
    return iguales

def rota(theta):
    a = np.cos(theta)
    b = np.sin(theta)
    A = np.array([[a,-b],[b,a]])
    return A

def escala(s):
    A = np.zeros((len(s),len(s)))
    for i in range(len(s)):
        A[i][i] = s[i]
    return A

def rota_y_escala(theta,s):
    T2 = escala(s)
    T1 = rota(theta)
    return multiplicarMatrices(T2, T1)

def afin(theta,s,b):
    T2 = escala(s)
    T1 = rota(theta)
    A = T2@T1
    B = np.zeros((3,3))
    for i in range(2):
        B[i][:2] = A[i]
        B[i][2] = b[i]
    B[2][2] = 1
    return B

def trans_afin(v,theta,s,b):
    A = afin(theta,s,b)
    v1 = np.array([v[0],v[1],1])
    w = A@v1
    return w[:2]

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

def normaliza(X,p):
    normas = []
    for i in range(len(X)):
        vector = X[i].astype(float).copy()
        for j in range(len(vector)):
            vector[j] = vector[j]/norma(X[i],p)
        normas.append(vector)
    return normas

def normaMatMC(A,q,p,Np):
    vectores = np.array([np.random.uniform(-1,1,size=len(A[0])) for n in range(Np)])
    normalizados = normaliza(vectores,p)
    max = []
    norm = 0
    for vector in normalizados:
        norma_matriz = norma(A@vector,q)
        if norma_matriz > norm:
            norm = norma_matriz
            max = vector
    return norm,max

def traspuesta(A):
    f,c = A.shape
    T = np.zeros((c,f))
    i = 0
    while i < c:
        T[i,:] = A[:,i]
        i+=1
    return T

def normaExacta(A,p=[1,"inf"]):
    if isinstance(p,int):
        if p == 1:
            norm = 0
            for j in range(len(A[0])):
                norma_columna = 0
                for i in range(len(A)):
                    norma_columna += abs(A[i][j])
                if norma_columna > norm:
                    norm = norma_columna
        else:
            norm = None
    elif isinstance(p,str):
        if p == "inf":
            norm = 0
            for fila in A:
                norma_fila = norma(fila,1)
                if norma_fila > norm:
                    norm = norma_fila
        else:
            norm = None
    else:
        if p[0] == 1:
            norm = [normaExacta(A,1),normaExacta(A,"inf")]
        else:
            norm = None
    return norm

def condMC(A,p,Np):
    A_inversa = np.linalg.inv(A)
    nro_cond = normaMatMC(A,p,p,Np)[0]*normaMatMC(A_inversa,p,p,Np)[0]
    return nro_cond

def condExacta(A,p):
    A_inversa = np.linalg.inv(A)
    nro_cond = normaExacta(A,p)*normaExacta(A_inversa,p)
    return nro_cond

def multiplicarMatrices(A,B):
    res = np.zeros((len(A),len(B[0])))
    for f in range(len(A)):
        for c in range(len(B[0])):
            sum = 0
            for fb in range(len(B)):
                sum += A[f][fb]*B[fb][c]
            res[f][c] = sum

    return res

def calculaLU(A):
    cant_op = 0

    if A is None:
        return None, None, cant_op
    m, n = A.shape
    Ac = A.copy().astype(float)
    
    if m!=n:
        return None,None,cant_op

    for col in range(n):
        pivo = Ac[col][col]
        if abs(pivo) < 1e-15:
            return None, None, 0

        for fil in range(col+1,m):
            if Ac[fil][col] != 0:
                Ac[fil][col] = Ac[fil][col]/pivo
                cant_op +=1

                Ac[fil][col+1:] -= Ac[col][col+1:]*(Ac[fil][col])

        cant_op += 2*((n - col-1)**2)

    L = np.eye(n)
    U = np.zeros((m,n))
    for f in range(m):
        for c in range(n):
            if f > c:
                L[f][c] = Ac[f][c]
            else:
                U[f][c] = Ac[f][c]
    
    return L, U, cant_op

def es_inferior(A):
    inferior = True
    for i in range(len(A)):
        for j in range(i+1,len(A[0])):
            inferior = inferior and A[i][j] == 0
    return inferior

def res_tri(L,b,inferior=True):
    x = np.zeros(len(L))
    if not inferior:
        for i in range(len(L)-1,-1,-1):
            sum = 0
            for j in range(len(L)-1,i,-1):
                sum += L[i][j]*x[j]
            x[i] = (b[i] - sum)/L[i][i]
        return x
    for i in range(len(L)):
        sum = 0
        for j in range(i):
            sum += L[i][j]*x[j]
        x[i] = (b[i] - sum)/L[i][i]
        
    return x

def inversa(A):
    L, U, nops = calculaLU(A)
    if L is None or U is None:
        return None
    L_inv = np.eye(len(L))
    for i in range(len(L[0])):
        piv = L[i][i]
        for j in range(i+1,len(L)):
            l = L[j][i]/piv
            L[j][:i+1] -= L[i][:i+1]*l
            L_inv[j][:i+1] -= L_inv[i][:i+1]*l

    U_inv = np.eye(len(U))
    for i in range(len(U[0])-1,-1,-1):
        if U[i][i] != 1:
            U_inv[i] = U_inv[i]/U[i][i]
            U[i] = U[i]/U[i][i]
        piv = U[i][i]
        for j in range(i):
            u = U[j][i]/piv
            U[j][i:] -= U[i][i:]*u
            U_inv[j][i:] -= U_inv[i][i:]*u

    return multiplicarMatrices(U_inv,L_inv)

def calculaLDV(A):
    L,U,nops = calculaLU(A)
    if L is None or U is None:
        return None
    U = traspuesta(U)
    V,D, nops = calculaLU(U)
    V = traspuesta(V)

    return L,D,V

def esSimetrica(A,tol=1e-10):
    f,c = A.shape
    if f == c:
        T = traspuesta(A)
        simetrica = True
        i = 0
        while simetrica and i < f:
            for j in range(c):
                simetrica = simetrica and np.isclose(A[i,j], T[i,j], tol)
            i += 1
    else:
        simetrica = False
    return simetrica

def esSDP(A, atol=1e-5):
    if not esSimetrica(A,atol):
        return False
    if calculaLDV(A) is None:
        return False
    L,D,V = calculaLDV(A)
    diag_pos = True
    for i in range(len(D)):
        diag_pos = diag_pos and D[i][i] > 0
    return diag_pos

def calculaCholesky(A,atol=1e-10): #mision cumplida
    if not esSDP(A,atol):
        return None
    L,D,V = calculaLDV(A)

    for i in range(len(D)):
        D[i][i] = np.sqrt(D[i][i])

    R = multiplicarMatrices(L,D)
    return R

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    if len(A) != len(A[0]):
        return None, None

    Q = np.zeros((len(A), len(A)))
    R = np.zeros((len(A), len(A)))
    A_tras = traspuesta(A)
    
    no_descartados = []
    for j in range(len(A)):
        qj = A_tras[j].copy()
        for k in no_descartados:
            R[k][j] = producto_interno(Q[k],qj)
            qj -= R[k][j]*Q[k]
        R[j][j] = norma(qj, 2)   
        if norma(qj, 2) > tol:
            Q[j] = qj/R[j][j]
            no_descartados.append(j)
    
    Q_mono = traspuesta(Q[no_descartados])
    R_mono = R[no_descartados,:]
    return Q_mono, R_mono

def QR_con_HH(A,tol=1e-12,extras=False):
    if len(A) < len(A[0]):
        return None, None
    
    R = A.copy().astype(float)
    Q = np.eye(len(A))
    for k in range(len(A)):
        x = np.zeros(len(A) - k)
        for f in range(len(A) - k):
            x[f] = R[f + k][k]
        a = -x[0]*norma(x, 2)/abs(x[0])
        u = np.array(x.copy())
        u[0] -= a 
        if norma(u, 2) > tol:
            u = u/norma(u, 2)
            Hk = np.eye(len(A)-k)
            Hk -= 2*multiplicarMatrices(u.reshape((len(u),1)), traspuesta(u.reshape((len(u),1))))
            Hk_mono = np.eye(len(A))
            Hk_mono[k:,k:] = Hk
            R = multiplicarMatrices(Hk_mono, R)
            Q = multiplicarMatrices(Q, traspuesta(Hk_mono))
    return Q, R

def calculaQR(A,metodo='RH',tol=1e-12):
    if metodo == 'RH':
        Q, R = QR_con_HH(A, tol)
    elif metodo == 'GS':
        Q, R = QR_con_GS(A,tol)
    else:
        return None, None
    return Q, R

def multiplicarAx(A,x):
    Ax = np.zeros(len(x))
    for i in range(len(A)):
        sum = 0
        for j in range(len(x)):
            sum += A[i][j]*x[j]
        Ax[i] = sum
    return Ax

def producto_interno(x, y):
    if len(x) != len(y):
        return None
    prod = 0
    for i in range(len(x)):
        prod += x[i]*y[i]
    return prod

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
    k = 1 #preguntar criterio
    while abs(e - 1)> tol and k < K:
        v = v_m
        v_m = f_A(A,v,2)
        e = producto_interno(v_m,v)
        k += 1
    Av_m = multiplicarAx(A, v_m)
    lamda = producto_interno(v_m, Av_m)
    error = e - 1
    return v_m, lamda, k

def diagRH(A,tol=1e-15,K=1000):
    if not esSimetrica(A, tol):
        return None, None
    v, lamda, k = metpot2k(A, tol, K)
    if lamda<tol:
        return np.zeros((len(A),len(A))), np.zeros((len(A),len(A)))
    I = np.eye(len(A))
    u = I[0].copy()
    u -= v
    norma_u = norma(u,2)
    if norma_u<tol:
        H = I
    else:
        H = I - (2/norma_u**2)*multiplicarMatrices(u.reshape((len(u),1)), traspuesta(u.reshape((len(u),1))))
    if len(A) == 2:
        S = H
        D = multiplicarMatrices(H, multiplicarMatrices(A, traspuesta(H)))
    else:
        B = multiplicarMatrices(H, multiplicarMatrices(A, traspuesta(H)))
        A_mon = B[1:, 1:]
        S_mon, D_mon = diagRH(A_mon, tol, K)
        D = np.zeros((len(A),len(A)))
        D[0][0] = lamda
        D[1:, 1:] = D_mon
        S1 = np.eye(len(A))
        S1[1:, 1:] = S_mon
        S = multiplicarMatrices(H, S1)    
    return S, D


'''
#con este test deberia ser none porque te dicen que GS esta definido solo para cuadradas
A = np.array([[1.0, 2.0, 0.0],[2.0, 1.0, 1.0],[0.0, 1.0, 2.0],[1.0, 0.0, 1.0],[3.0, 1.0, 0.0]])
Q, R = calculaQR(A, metodo='GS')
if Q is not None:
   print('Q ok')
else:
   print('Problema con Q, da None')
if R is not None:
   print('R ok')
else:
   print('Problema con R, da None')
if R.shape == (2, 3):
   print('R tiene forma (2, 3)')
else:
   print('R deberia tener forma (2, 3) y tiene forma ' + str(R.shape) + '')
if Q.shape == (3, 2):
   print('Q tiene forma (3, 2)')
else:
   print('Q deberia tener forma (3, 2) y tiene forma ' + str(Q.shape) + '')
triangular = True
for i in range(R.shape[0]):
   for j in range(min(i, R.shape[1])):
      if abs(R[i, j]) >= 1e-4:
         triangular = False
if triangular:
   print('R es triangular superior')
else:
   print('R no es triangular superior')
reconstructed = Q @ R
if np.allclose(reconstructed, A, atol=1e-3):
    print('Q@R reconstruye A')
else:
   print('Q@R no reconstruye A')
QtQ = traspuesta(Q) @ Q
if np.allclose(QtQ, np.eye(Q.shape[1]), atol=1e-5):
   print('QtQ es la identidad')
else:
   print('QtQ no da identidad')

print(Q@R)
print(Q)
print(R)
print(A)
'''