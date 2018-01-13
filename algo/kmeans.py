import numpy
import scipy.spatial as spatial
import pylab

global M

def ppcentroide(k, assign_0):
    global M
    
    n = len(M)

    # assign[i] = k ssi le point i est dans le cluster k
    assign = numpy.zeros((n, ), dtype=numpy.int32)

    # Dist_k[k] = inertie interne au cluster k
    # Size_k[k] = taille du cluster k
    Dist_k = numpy.zeros((k, ))
    Size_k = numpy.zeros((k, ), dtype=numpy.int32)

    # Calcul des deux quantites definies ci-dessus
    for i in range(n):
        i_k = assign_0[i]
        Size_k[i_k] += 1
        Dist_k[i_k] += M[i][i]
        for j in range(i+1,n):
            if i_k == assign_0[j]:
                Dist_k[i_k] += 2*M[i][j]
    for i in range(k):
        Dist_k[i] /= Size_k[i]**2

    # Calcul des nouvelles assignations
    for i in range(n):
        # d_min = "distance" de i au cluster auquel il est assigne pour le moment
        d_min = numpy.inf
        # Pour chaque cluster k...
        for i_k in range(k):
            # d_k = "distance" de i au cluster k
            d_k = 0

            d_tmp = 0
            for j in range(n):
                if assign_0[j] == i_k:
                    d_tmp += M[i][j]
            d_k += -2*d_tmp/Size_k[i_k]

            d_k += M[i][i]

            d_k += Dist_k[i_k]

            if d_k <= d_min:
                assign[i] = i_k
                d_min = d_k

    return assign


def kmeans(mat_dist, k, n_iter):
    global M
    M = mat_dist
    n = len(M)

    # Assignation aleatoire des points aux clusters
    assign = [numpy.random.randint(0,k) for i in range(n)]
    for i in range(k):
        assign[i] = i # Pour etre sur d'avoir au moins 1 point par cluster

    for iter in range(n_iter):
        assign = ppcentroide(k, assign)
    assign = ppcentroide(k, assign)
    return assign

# Zone d'interface
def do(mat_dist, k, n_iter):
    assign =  kmeans(mat_dist, k, n_iter)
    return assign, k

