# FUNCIONES PEDIDAS EN EL TP
import numpy as np
import pandas as pd
from alc import diagRH, multiplicarMatrices, calculaQR, inversa, multiplicarAx, norma, calculaCholesky

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

