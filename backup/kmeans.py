import numpy
import scipy.spatial as spatial
import pylab

# Calcul le cout du clustering courant
def compute_cost(X, C, assign):
    n = len(X)
    d = len(X[0])
    acc = 0.
    for i in range(n):
        acc += numpy.linalg.norm(X[i] - C[assign[i]]) ** 2
    return acc

# Assigne à chaque point de X le n° du point de C dont il est le plus proche 
def ppcentroide(X, C):
    mat_dist = spatial.distance.cdist(X, C, "euclidean")
    return numpy.argmin(mat_dist, axis=1)

# Algorithme kmeans
def kmeans(X, k, n_iter):
    n = len(X)
    d = len(X[0])
    
    cost = numpy.inf
    
    A = [i for i in range(n)]
    B = []
    for i in range(k):
        B.append(A.pop(numpy.random.randint(0,n-i)))
    C = [X[i] for i in B]

    for iter in range(n_iter):
        assign = ppcentroide(X, C)
        for i_k in range(k):
            cluster = [X[i] for i in range(n) if assign[i] == i_k]
            C[i_k] = numpy.mean(cluster, axis = 0)
        cost = compute_cost(X, C, assign)
        print(iter, cost)
    assign = ppcentroide(X, C)
    return C, assign, cost

# Zone d'interface


# Zone de tests

###
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
###

filename = 'cinq_classes1'

X = []
read_data('./data/' + filename + '/' + filename + '.txt',X)

s = 2574927556214
n = len(X)
k = 5
n_iter = 30


numpy.random.seed(s)
C, assign, cost = kmeans(X, k, n_iter)

pylab.figure()
colors = "rbgkmc"
for i_k in range(k):
    pylab.plot([X[i][0] for i in range(n) if assign[i] == i_k],
               [X[i][1] for i in range(n) if assign[i] == i_k],
               colors[i_k] + ".")
    pylab.plot(C[i_k][0], C[i_k][1], colors[i_k] + "o")    
pylab.show()

write_cluster('./data/' + filename + '/kmeans.txt', assign, k, n_iter, s)
