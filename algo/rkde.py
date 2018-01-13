import numpy as np
import sklearn.metrics.pairwise as skmp

''' Parametres globaux '''

# Parametres globaux :
# - parametres du noyau
# - distribution courante
# - poids courants (pour l'estimation robuste)
# - parametres de l'estimateur robuste
M = np.array([])
distribution = np.array([])
weights = np.array([])
precision = 10
cap = 0.
epsilon = 0.00001

''' Fonctions generales '''

# Estimateur robuste
def huber(x):
    global cap
    if x < cap:
        return 1.
    else:
        return cap / x

def huber_array(X):
    return np.array([huber(x) for x in X])

def linear(x):
    global epsilon
    return 1. / np.sqrt(np.power(x, 2.) + np.power(epsilon, 2.))

''' Manipulation de la distribution '''

# Reinitialisation de la distribution courante
def rkde_renew():
    global distribution
    global weights
    distribution = np.array([])
    weights = np.array([])

# Initialisation de la matrice des distances
def rkde_init(mat_dist, bandwidth):
    global M
    global cap
    M = mat_dist
    cap = 2. * bandwidth

def rkde_eval_array(X):
    global distribution
    global weights
    n = len(weights)
    m = len(X)
    #return np.sum(np.reshape(np.tile(weights, m), (m, n)) * M[X, distribution], axis = 1)
    return np.sum(np.tile(weights, m).reshape((m, n)) * M[np.ix_(X, distribution)], axis = 1)

# Estimation de la densite d'un ensemble de points donne
def rkde_fit(cluster):
    global distribution
    global weights
    global precision
    distribution = np.array([np.array(x) for x in cluster])

    n = len(distribution)
    weights = (1. / n) * np.ones(n)

    for k in range(precision):
        a_const = np.sum(weights * rkde_eval_array(distribution))
        tmp = M[np.ix_(distribution,distribution)]
        tmp_d = np.diag(tmp)
        a = huber_array(np.sqrt(np.absolute( 
            tmp_d
            + a_const * np.ones(n)
            - 2. * np.inner(weights, tmp) )))
        s = np.sum(a)
        weights = a / s

# Evaluation d'un vecteur de points suivant la distribution courante
def rkde_score_samples(points):
    return rkde_eval_array(np.array(points))

''' Clustering '''

# Entree :
# - mat_dist : matrice des distances entre les points
# - bandwidth : flottant
# - prob : flottant
# Sortie :
# - assign : Vecteur de N entiers. Assignation de chaque element de X a un cluster.
# - nb_cluster : Entier. Nombre de clusters crees
def cluster_rkde(mat_dist, bandwidth, prob):
    rkde_init(np.array(mat_dist),bandwidth)
    n = len(mat_dist)
    # A chaque point on associera un cluster
    assign = [0 for i in range(n)]
    # Y est la liste des points non encore affectes a un cluster
    Y = list(range(n))
    
    # On demarre l'affectation des points
    n_cluster = 0
    while(len(Y) > 0):
        # Creation d'un cluster a partir d'un point au hasard
        rkde_renew()
        i = np.random.randint(0,len(Y))
        cluster = [Y[i]]
        assign[Y[i]] = n_cluster
        Y.pop(i)

        # Tant qu'on peut rajouter des points, on le fait
        wip = (len(Y) > 0)
        while wip:
            rkde_fit(cluster)
            Z = rkde_score_samples(Y)
            i = np.argmax(Z)
            if Z[i] > prob:
                # On a trouve un point convenable, on l'ajoute au cluster
                cluster.append(Y[i])
                assign[Y[i]] = n_cluster
                Y.pop(i)
                if (len(Y) == 0):
                    wip = 0
            else:
                wip = 0
        n_cluster += 1
        
    # On retourne assign
    return assign, n_cluster

''' Interface '''

# Zone d'interface
def do(mat_dist, bandwidth, prob):
    assign, nb_clusters = cluster_rkde(mat_dist, bandwidth, prob)
    return assign, nb_clusters
