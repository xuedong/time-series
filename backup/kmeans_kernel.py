import numpy
import scipy.spatial as spatial
import pylab

sigma = 10.

def K(Xi, Xj, s):
    return numpy.exp( - ( (numpy.linalg.norm(Xi - Xj) ** 2) / (2 * (s**2)) ) )

# Entrée :
#  - X l'ensemble des points
#  - k le nombre de clusters à former
#  - assign_0 l'assignation courante des points à un cluster
#  - M la matrice des distances entre les points
# Sortie :
#  - assign la nouvelle assignation des points à un cluster
#  - cost la fonction de cout (doit etre décroissante)
def ppcentroide(X, k, assign_0, M):
    global sigma
    n = len(X)
    cost = 0.

    # assign[i] = k ssi le point i est dans le cluster k
    assign = numpy.zeros((n, ), dtype=numpy.int32)

    # Dist_k[k] = inertie interne au cluster k
    # Size_k[k] = taille du cluster k
    Dist_k = numpy.zeros((k, ))
    Size_k = numpy.zeros((k, ), dtype=numpy.int32)

    #    # Calcul des deux quantités définies ci-dessus
    #    for i_k in range(k):
    #        d_tmp = 0
    #        C_size = 0
    #        for j1 in range(n):
    #            if (assign_0[j1] == i_k):
    #                C_size +=1
    #                for j2 in range(n):
    #                    if (assign_0[j2] == i_k):
    #                        d_tmp += M[j1][j2]
    #        Size_k[i_k] = C_size
    #        Dist_k[i_k] = d_tmp/C_size**2

    # Calcul des deux quantités définies ci-dessus
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
        # d_min = "distance" de i au cluster auquel il est assigné pour le moment
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
        cost += d_min

    return assign,cost


def kmeans(X, k, n_iter):
    n = len(X)
    cost = numpy.inf

    # Pré-calcul des distances entre les clusters
    M = [[K(X[i], X[j], sigma) for j in range(n)] for i in range(n)]
    

    # Assignation aléatoire des points aux clusters
    assign = [numpy.random.randint(0,k) for i in range(n)]
    for i in range(k):
        assign[i] = i # Pour etre sur d'avoir au moins 1 point par cluster

    for iter in range(n_iter):
        assign,cost = ppcentroide(X, k, assign, M)
        print(iter, cost)
    assign,cost = ppcentroide(X, k, assign, M)
    return assign,cost

### Zone de tests

# Fonctions utiles
def read_data(file_name, X):
    f = open(file_name, 'r')
    data = f.readlines()
    for line in data:
        X.append((map(float, filter(lambda x: x <> '\n', line.split("\t")))))
    f.close()

def write_cluster(file_name, assign, n_cluster, n_iter, s):
    n = len(assign)
    f = open(file_name, 'w')
    f.write("n_cluster = " + str(n_cluster) + "\n")
    f.write("n_iter = " + str(n_iter) + "\n")
    f.write("n_data = " + str(n) + "\n")
    f.write("seed = " + str(s) + "\n")
    for i in range(n):
        f.write(str(assign[i]) + "\n")
    f.close()

# Vrais tests

f = 'cinq_classes1'

X = []
read_data('./data/' + f + '/' + f + '.txt',X)

s = 0
n = len(X)
k = 5
n_iter = 10

numpy.random.seed(s)
assign,cost = kmeans(numpy.array(X), k, n_iter)

pylab.figure()
colors = "rbgkmc"
for i_k in range(k):
    pylab.plot([X[i][0] for i in range(n) if assign[i] == i_k],
               [X[i][1] for i in range(n) if assign[i] == i_k],
               colors[i_k] + ".")  
pylab.show()

write_cluster('./data/' + f + '/' + 'kernel' + '.txt', assign, k, n_iter, s)

