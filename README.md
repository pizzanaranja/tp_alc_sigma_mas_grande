# tp_alc_sigma_mas_grande
(X, Y Z) son las coordenadas de la camara
F es el centro de proyección de la camara
G plano imagen paralelo al plano XY, con ejes (u, v). Es un grilla rectangular de pixeles
(U, V, W) es el eje coordenado del mundo real.

Punto Mi del espacio real 
mi en el plano imagen, es la proyección de Mi sobre G
M0 es el origen del plano real (X0, Y0, Z0)

EJERCICIO 1
calib3d: detected corners tiene 210 filas, que suponemos que corresponden a un IMgen cada una
tiene 24 columnas, que suponemos son las coordenadas u y v por 12 puntos
pw corners: coordenadas en el mundo real de esos 12 puntos

Matriz Q
- Solución no única a menos de un factor de escala: queremos que todas kas soluciones sean multiplos de un mismo vector (una recta de soluciones)
- Qm = 0, es decir, el nucelo de Q debe tener dimension 1 (para que sea una recta)
. Rg(Q) = 12 - dim(Nu(Q)) = 11
- Como cada punto de Q ocupa 2 filas, necesitamos minimo 6 puntos para que tenga rango 12
. Los puntos no pueden ser coplanares: son necesarios 2 planos mínimo para que no se anulen columnas, es decir, mínimo 2 puntos en otro plano (4 en uno y 2 en otro es lo mínimo)
- Q se arma con la info de detected corners (u, v) y pw corners (U, V W)
- Para m hacemos núcleo de Q usando la función armada del labo (no tiene restricciones de tamaño)