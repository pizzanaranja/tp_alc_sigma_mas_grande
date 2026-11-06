import numpy as np
import pandas as pd


# MODULOS
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
                simetrica = simetrica and np.isclose(A[i,j],T[i,j],tol) 
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

def calculaCholesky(A,atol=1e-10): 
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
    v = np.random.uniform(-1,1,len(A))
    v_m = f_A(A,v,2)
    e = producto_interno(v_m, v)
    k = 1 #preguntar criterio!!
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
    v, lamda, _ = metpot2k(A, tol, K)
    if abs(lamda)<tol:
        return np.eye(len(A)), np.zeros((len(A), len(A)))
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


## FUNCIONES TP
# EJERCICIO 2
def obtenerQ3d(imgPts,objPts):
    #imgPts es un array que representa 1 sola foto
    #objPts es la matriz de coordenadas U,V,W de varios puntos en el mundo real
    n = len(objPts)
    Q = np.zeros((2*n,12))
    for i in range(n):
        #fila 1 del punto (coordenadas u)
        Q[2*i,:3] = objPts[i]
        Q[2*i,3] = 1
        u_i,v_i = imgPts[2*i], imgPts[2*i+1]
        Q[2*i,8:11] = -u_i*objPts[i]
        Q[2*i,11] = -u_i
        #fila 2 del mismo punto (coordenadas v)
        Q[2*i+1,4:7] = objPts[i]
        Q[2*i+1,7] = 1
        Q[2*i+1, 8:11] = -v_i*objPts[i]
        Q[2*i+1,11] = -v_i
    return Q

def resolverDLT3D(Q):
    S, _ = diagRH(multiplicarMatrices(Q.T,Q),1e-10)
    m = S[:,11] # 12x1, M 3x4
    M = m.reshape(3,4)
    return M


## el determinante de R tiene que ser 1 para que R sea matriz de rotacion, si da -1 es una rotoreflexion (pose imposible para una camara)
## pensamos poner pops como una lista con esos requisitos y leerla adentro de la funcion
def devolverMatricesCalibracion3d(M, condiciones_opc = None): #chequear este parametro, pueden ser 2 parametros, etc. Libertad. 
    """
    condiciones_opc:
    None = sin restricciones extra
    alpha = impone alpha_u = alpha_v
    skew = impone K[1,2] = 0
    ambas = impone alpha + skew
    """
    KR = M[:, :3]
    Kt = M[:, 3]
    Q, R_mono = calculaQR(inversa(KR))
    R = Q.T
    K = inversa(R_mono)
    for i in range(3): #nos aseguramos diagonal>0 en K
        if K[i,i]<0:
            K[:,i] = -K[:,i] #columna de K
            #una vez que cambiamos esto, para que den las cuentas tambien hay que 
            #cambiarlo en R. KR hace fila de K por columna de R. 
            R[i,:] = -R[i,:] #fila de R
    #elem = K[2,2]
    if condiciones_opc == "ambas":
        K[0,1] = 0
        prom = (K[0,0] + K[1,1])/2
        K[0,0] = prom
        K[1,1] = prom
    elif condiciones_opc == "alpha" : 
        prom = (K[0,0] + K[1,1])/2
        K[0,0] = prom
        K[1,1] = prom
    elif condiciones_opc == "skew":
        K[0,1] = 0 #chequear si esta bien imponer esto asi nomas o se hace modificando M!!!!!!!
    
    t = multiplicarAx(inversa(K), Kt)
    
    K[:, :] = K[:, :]/K[2,2] #normalizamos K
    #chequeamos si me quedó la versión esperada o espejada, según el signo del vector aleatorio (random uniform) usado en matpot2k para generar M, queremos que R sea de rotacion, o sea con det=1
    det_R = determinante3x3(R)
    if det_R < 0: #caso R reflexion, corregimos para que sea rotacion
        R = -R
        t = -t
        #elem = -elem
    return R, t, K

def determinante3x3(A):
    #desarrollando por la primera fila:
    det_A = (A[0,0]*(A[1,1]*A[2,2] - A[1,2]*A[2,1])
          - A[0,1]*(A[1,0]*A[2,2] - A[1,2]*A[2,0])
          + A[0,2]*(A[1,0]*A[2,1] - A[1,1]*A[2,0]))
    return det_A


# EJERCICIO 3
def obtenerQ2d(imgPts,objPts):
    n = len(objPts)
    Q = np.zeros((2*n,9))
    for i in range(n):
        #fila 1 del punto (coordenadas u)
        Q[2*i,:2] = objPts[i]
        Q[2*i,2] = 1
        u_i,v_i = imgPts[2*i], imgPts[2*i+1]
        Q[2*i,6:8] = -u_i*objPts[i]
        Q[2*i,8] = -u_i
        #fila 2 del mismo punto (coordenadas v)
        Q[2*i+1,3:5] = objPts[i]
        Q[2*i+1,5] = 1
        Q[2*i+1, 6:8] = -v_i*objPts[i]
        Q[2*i+1,8] = -v_i
    return Q

def armar_Hs (img_pts,obj_pts):
    Hlst = []
    for i in img_pts:
        H = estimar_H(i, obj_pts)
        Hlst.append(H)
    return Hlst

def estimar_H(vista,obj_pts):
    # 1. Matriz Q de la vista actual
    Q = obtenerQ2d(vista, obj_pts)

    # 2. Resolver Qh = 0 con autovalores/SVD de ALC
    QtQ = multiplicarMatrices(Q.T, Q)
    S, D = diagRH(QtQ, 1e-08)

    h = S[:, -1]  # Autovector del menor autovalor
    H = h.reshape(3, 3)
    H = H / H[2, 2]  # Normalizar escala
    return H

def armarV(x, y):
    v = np.array([x[0]*y[0],
                  x[0]*y[1] + y[0]*x[1],
                  x[1]*y[1],
                  x[0]*y[2] + y[0]*x[2],
                  x[1]*y[2] + y[1]*x[2],
                  x[2]*y[2]])
    return v

def armarA(Hlst):
    f = len(Hlst)
    A = np.zeros((2*f, 6))

    for i in range(f):
        h1 = Hlst[i][:,0]
        h2 = Hlst[i][:,1]
        a1 = armarV(h1,h2)
        a2 = armarV(h1,h1) - armarV(h2,h2)
        A[2*i] = a1
        A[2*i + 1] = a2
    return A

def calcularMatrizK2D(A):
    S, _ = diagRH(multiplicarMatrices(A.T, A))
    b = S[:, 5] #A es de 2nx6, AtA es de 6x6
    
    B = np.array([[b[0], b[1], b[3]],
                [b[1], b[2], b[4]],
                [b[3], b[4], b[5]]])

    #para que sea definida positiva la diagonal tiene que serlo
    if B[0][0] < 0:
        B = -B

    L = calculaCholesky(B)
    K = inversa(L.T)
    K = K/K[2,2]    
    return K

def productoCruz(x,y):
    v = np.array([x[1]*y[2] - x[2]*y[1],
                  x[2]*y[0] - x[0]*y[2],
                  x[0]*y[1] - x[1]*y[0]])
    return v

def obtenerPose2D(Hj, K):
    K_inv = inversa(K)
    K_invH = multiplicarMatrices(K_inv, Hj)
    columnasKiH = K_invH.T
    lamda = 1/norma(columnasKiH[0],2)
    
    r1 = lamda*columnasKiH[0]
    r2 = lamda*columnasKiH[1]

    tj = lamda*columnasKiH[2]

    #la imagen puede generarse detras de la camara pero tenemos que
    #generarla adelante, por eso le cambiamos el signo
    if tj[2] < 0:
        r1, r2, tj = -r1, -r2, -tj
    r3 = productoCruz(r1, r2)
    Rj = np.array([r1, r2, r3]).T

    return Rj, tj

## EJERCICIO 4
def circulo (centro, radio = 1.0, n= 100):
    angulos = 2*np.pi*np.arange(n+1)/n
    U = centro[0] + radio*np.cos(angulos)
    V = centro[1] + radio*np.sin(angulos)
    return U,V #devuelve lista de coordenadas de los puntos del circulo en mundo real

## EJERCICIO 5
def recuperarPares(csv_path):
    print(csv_path)
    df = pd.read_csv(csv_path, sep=';', engine="python")
    object_single = df[["X", "Y", "Z"]].to_numpy(dtype=np.float32)

    obj_points = []
    img_points = []

    for i in range(4):
        image_single = df[[f"u_{i}", f"v_{i}"]].to_numpy(dtype=np.float32)

        obj_points.append(object_single.copy())
        img_points.append(image_single.copy())

    return obj_points, img_points

def obtener_matriz_normalizacion(pts):
    """
    Calcula la matriz de normalización isotrópica 3x3 para un conjunto de puntos 2D.
    Centra los puntos en el origen y escala para que la distancia RMS al origen sea sqrt(2).
    """
    pts = np.array(pts).reshape(-1, 2)
    n = len(pts)
    
    # 1. Centroide de los puntos
    media = np.mean(pts, axis=0)
    
    # 2. Trasladar al origen y calcular distancia media RMS
    pts_centrados = pts - media
    distancias = np.sqrt(np.sum(pts_centrados**2, axis=1))
    media_distancia = np.mean(distancias)
    
    if media_distancia == 0:
        s = 1.0
    else:
        s = np.sqrt(2) / media_distancia
    
    # 3. Construir matriz T de normalización (3x3)
    T = np.array([
        [s, 0, -s * media[0]],
        [0, s, -s * media[1]],
        [0, 0, 1]
    ], dtype=float)
    
    return T

def pasar_a_real(pxs, Hinv):
    puntos = []

    for pixel in pxs:
        pix = [pixel[0], pixel[1], 1]
        punto = multiplicarAx(Hinv, pix)
        punto = punto/punto[2]
        puntos.append(punto)
    
    return puntos

def calcularVelocidades(posiciones, tiempos):
    velocidades = []
    for i in range(len(posiciones)-1):

        # Diferencia de posiciones
        desplazamiento = posiciones[i+1] - posiciones[i]

        # Distancia recorrida en metros
        distancia = np.linalg.norm(desplazamiento)

        # Diferencia temporal en segundos
        dt = tiempos[i+1] - tiempos[i]

        # Velocidad en m/s
        velocidad = distancia / dt

        velocidades.append(velocidad)

    return np.array(velocidades)