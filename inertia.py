import numpy as np

def center(C):
    n = len(C)
    d = len(C[0])

    # Le centre du cluster est A
    A = [0 for i in range(d)]
    for i in range(n):
        for j in range(d):
            A[j] += C[i][j]
    return A

def intra(X, assign):
    X = np.array(X)
    n = len(X)
    k = max(assign) + 1
    # Formation des k clusters
    clusters = [ np.array([X[i] for i in range(n) if assign[i] == j]) for j in range(k)]
    # Calcul du centre des clusters
    mu = [center(clusters[i]) for i in range(k)]
    # On centre les points des clusters
    clusters = [ [
        (clusters[j][i] - mu[j])
        for i in range(len(clusters[j])) ]
        for j in range(k) ]
    # On prend la norme au carré des points des clusters
    clusters = [ [
        np.linalg.norm(clusters[j][i])**2
        for i in range(len(clusters[j])) ]
        for j in range(k) ]
    # On somme ces normes, on renvoie tout
    return sum( [
        sum(clusters[i])
        for i in range(len(clusters)) ] )

def inter(X, assign):
    n = len(X)
    k = max(assign) + 1
    c = np.array(center(X))
    clusters = [ np.array([X[i] for i in range(n) if assign[i] == j]) for j in range(k) ]
    sizes = [ len(clusters[i]) for i in range(k) ]
    mu = [ center(clusters[i]) for i in range(k) ]
    mu = [ mu[i] - c for i in range(k) ]
    costs = [
        sizes[i] * np.linalg.norm(mu[i])**2 
        for i in range(k) ]
    return sum(costs)



## Zone de tests

X = [
    [0,0],[0,1],[1,0],[1,1],
    [8,8],[8,9],[9,8],[9,9]
]
assign = [0,0,0,0,1,1,1,1]

print(intra(X,assign))
print(inter(X,assign))
