import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Paramètres :
# -> X : Vecteur de N point de dimension D
# -> bandwidth : flottant
# -> prob : flottant
# # # # # # #
# Sortie :
# -> assign : Vecteur de N entiers. Assignation de chaque élément de X à un cluster.
# nb_cluster : Entier. Nombre de clusters créés
def cluster_kde(X, bandwidth=1, prob=0.01):
    n = len(X)
    # A chaque point on associera un cluster
    assign = [0 for i in range(n)]
    # Y est une duplication de X, où à chaque élément on associe sa position dans X
    # Cela servira à remplir assign
    Y = [[X[i],i] for i in range(n)]

    # On démarre l'affectation des points
    n_cluster = 0
    while(len(Y) > 0):
        # Création d'un cluster
        kde = KernelDensity(bandwidth=bandwidth)
        cluster = [Y[0][0]]
        assign[Y[0][1]] = n_cluster
        Y.pop(0)
        
        # Tant qu'on peut rajouter des points, on le fait
        wip = len(Y) > 0
        while wip:
            kde.fit(cluster)
            Z = [Y[i][0] for i in range(len(Y))]
            Z = kde.score_samples(Z)
            i = np.argmax(Z)
            if np.exp(Z[i]) > prob:
                # On a trouvé un point convenable, on l'ajoute au cluster
                cluster.append(Y[i][0])
                assign[Y[i][1]] = n_cluster
                Y.pop(i)
                wip = len(Y) > 0
            else:
                wip = 0
        n_cluster += 1

    # On retourne assign
    return assign, (n_cluster+1)

# Fonctions utiles
def read_data(file_name, X):
    f = open(file_name, 'r')
    data = f.readlines()
    for line in data:
        X.append((map(float, filter(lambda x: x <> '\n', line.split("\t")))))
    f.close()

def write_cluster(file_name, assign, n_cluster, bandwidth, prob, s):
    n = len(assign)
    f = open(file_name, 'w')
    f.write("n_cluster = " + str(n_cluster) + "\n")
    f.write("bandwidth = " + str(bandwidth) + "\n")
    f.write("prob = " + str(prob) + "\n")
    f.write("seed = " + str(s) + "\n")
    for i in range(n):
        f.write(str(assign[i]) + "\n")
    f.close()


# Zone de tests

f = 'deux_classes'

X = []
read_data('./data/' + f + '/' + f + '.txt',X)

s = 0
n = len(X)
bandwidth = 10
prob = 0.001

np.random.seed(s)
assign,n_cluster = cluster_kde(X, bandwidth, prob)

plt.figure()
colors = "bgrcmyk"
symbols = ".ov18sp*h+xD_"
for i_k in range(n_cluster):
    plt.plot( [X[i][0] for i in range(n) if assign[i] == i_k],
              [X[i][1] for i in range(n) if assign[i] == i_k],
              colors[i_k % 7] + symbols[i_k / 7] )
plt.show()

write_cluster('./data/' + f + '/' + 'kde' + '.txt', assign, n_cluster, bandwidth, prob, s)
