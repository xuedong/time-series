try:
    import sys
    import numpy as np
    import scipy as sp
    import pylab
    import global_align as ga
    import random
    import marshal
    import matplotlib.pyplot as plt
except:
    exit(1)

import main

import kappa

""" Fonctions annexes """

def bandwidth_estimator(M):
    n = len(M)
    s = np.sqrt(np.sum(np.power(np.mean(M,0),2)))
    return 1.06 * s * (np.power(n,-.2))

def load_time_series_modis_GX():
    max_item = 20 # nb d'items lus au plus par classe
    file_path = '../data/modis_GX/'
    file_names = ['classe_bati.txt',
                  'classe_bois.txt',
                  'classe_cultures.txt',
                  'classe_eau.txt',
                  'classe_prairies.txt']
    data = np.array([ np.array([np.reshape(np.array(d),(len(d),1))
                                for d in main.read_data(file_path + file_name, max_item)])
                      for file_name in file_names])
    assign = np.concatenate( [i*np.ones(len(list(data[i]))) for i in range(len(data))] )
    data = np.concatenate(data)
    return data, assign

def load_time_series_ucr(data_name):
    max_item = -1 # nb d'items lus au plus par classe
    file_path = '../data/ucr/'
    file_name = data_name + '/' + data_name + '_TRAIN'
    data = main.read_data(file_path + file_name, max_item)
    assign = np.array( [int(d[0]) for d in data ] )
    data = [d[1:] for d in data]
    data = [np.array(d) for d in data]
    data = [np.reshape(d,(len(d),1)) for d in data]
    data = np.array(data)
    return data, assign

def compute_mat(data, sigma, t, verbose=True):
    if verbose:
        n = len(data)
        m_0 = np.zeros((n,n))
        for i in range(n):
            print(str( (100. * i) / n ) + '%')
            for j in range(n):
                m_0[i,j] = np.exp( - ga.tga_dissimilarity(data[i], data[j], 2*sigma, t) )
    else:
        m_0 = np.array([[np.exp( - ga.tga_dissimilarity(d_1, d_2, 2*sigma, t) )
                         for d_1 in data]
                        for d_2 in data])
    return m_0

# A mieux utiliser
def search(goal, data, method, seed, f):
    sigma = 16.
    sigma_inf = 0.
    sigma_sup = 0. # dummy value
    cutoff = .005
    cutoff_inf = 0.
    cutoff_sup = 1.
    assign = []
    nb_c = -1
    while(nb_c != goal):
        ##################
        random.seed(seed)
        m_0 = f(sigma)
        assign,nb_c = main.make_clusters(m_0, method, [sigma, cutoff])
        if (nb_c < goal):
            # sigma must be <
            sigma_sup = sigma
            sigma = (sigma_inf + sigma) / 2.
            # cutoff must be >
            cutoff_inf = cutoff
            cutoff = (cutoff + cutoff_sup) / 2.
        else:
            # sigma must be >
            sigma_inf = sigma
            if (sigma_sup == 0.):
                sigma *= 2
            else:
                sigma = (sigma + sigma_sup) / 2.
            # cutoff must be <
            cutoff_sup = cutoff
            cutoff = (cutoff + cutoff_inf) / 2.
        print('sigma=' + str(sigma) + ', nb_c=' + str(nb_c) + ', cutoff=' + str(cutoff))
    return sigma, cutoff, assign


def remove_outliers(assign, cutoff):
    ind = list(set(assign))
    ind_tmp = max(ind) + 1 # indice correspondant aux outliers
    for i in ind:
        nb_occ = len(assign) - np.count_nonzero(assign - i)
        if (nb_occ <= cutoff):
            assign = [ind_tmp if j == i else j for j in assign]
    ind = list(set(assign))
    assign = [ind.index(i) for i in assign]
    return assign

def compute_median(data):
    d = len(data[0][0]) # Dimension des points
    data_f = data.ravel() # Redimensionnement des donnees (mise a plat)
    data_f = data_f.reshape((len(data_f)/d,d)) # Redimensionnement des donnees (ensemble des points)
    distances = sp.spatial.distance_matrix(data_f, data_f) # ensemble des distances entre les points
    return np.median(distances, overwrite_input=True) # Medianne des distances (pas besoin de redimensionner)

""" Zone de tests """

data_name = 'synthetic_control'

"""

# Chargement des donnees
print('Loading data...')
data, assign = load_time_series_ucr(data_name)
print('done\n')

# Pre-traitements
print('Pre-processing...')
assign = remove_outliers(assign, 1)
nb_class = max(assign) + 1
main.write_cluster('../result/ucr/' + data_name + '/original', assign, -1, -1)

med_dist = compute_median(data)
print('done\n')


# RKDE
seed = 2497224565
t = 5
print('rkde clustering...')
sigma_tga, cutoff_tga, assign_rkde_tga = search(
    nb_class, data, 'rkde', seed,
    lambda sigma: compute_mat(data, sigma, t))
sigma_gak, cutoff_gak, assign_rkde_gak = search(
    nb_class, data, 'rkde', seed,
    lambda sigma: compute_mat(data, sigma, 0))
sigma_gau, cutoff_gau, assign_rkde_gau = search(
    nb_class, data, 'rkde', seed,
    lambda sigma: sp.spatial.distance.squareform( sp.spatial.distance.pdist(
        data.reshape(  (len(data),len(data[0]))  ),
        lambda x,y: np.exp( - np.power(np.linalg.norm(x-y),2) / np.power(sigma,2))) ))
print('done\n')

# K-means
seed = 485321
nb_iter = 10
t = 5
sigma = 2 * med_dist
print('Starting kernel k-means clustering...')
random.seed(seed)
m_tga = compute_mat(data, sigma, t)
assign_kmeans_tga, nb_c_kmeans_tga = main.make_clusters(m_tga, 'kmeans', [nb_class, nb_iter])
random.seed(seed)
m_gak = compute_mat(data, sigma, 0)
assign_kmeans_gak, nb_c_kmeans_gak = main.make_clusters(m_gak, 'kmeans', [nb_class, nb_iter])
random.seed(seed)
m_gau = sp.spatial.distance.squareform( sp.spatial.distance.pdist(
    data.reshape(  (len(data),len(data[0]))  ),
    lambda x,y: np.exp( - np.power(np.linalg.norm(x-y),2) / np.power(sigma,2))) )
assign_kmeans_gau, nb_c_kmeans_gau = main.make_clusters(m_tga, 'kmeans', [nb_class, nb_iter])
print('done\n')

# Writing results
print('Writing results')
#main.write_cluster('../result/ucr/' + data_name + '/rkde_tga', assign_rkde_tga, -1, seed)
#main.write_cluster('../result/ucr/' + data_name + '/rkde_gak', assign_rkde_gak, -1, seed)
#main.write_cluster('../result/ucr/' + data_name + '/rkde_gau', assign_rkde_gau, -1, seed)
main.write_cluster('../result/ucr/' + data_name + '/kmeans_tga', assign_kmeans_tga, nb_iter, seed)
main.write_cluster('../result/ucr/' + data_name + '/kmeans_gak', assign_kmeans_gak, nb_iter, seed)
main.write_cluster('../result/ucr/' + data_name + '/kmeans_gau', assign_kmeans_gau, nb_iter, seed)
print('done\n')

# Calculs de kappa
print('Starting computing kappas...')
n_tests = 8
assign_tab = []
assign_original = main.read_assign('../result/ucr/' + data_name + '/original')
assign_tab.append(assign_original)
assign_rkde_tga = main.read_assign('../result/ucr/' + data_name + '/rkde_tga')
assign_tab.append(assign_rkde_tga)
assign_rkde_gak = main.read_assign('../result/ucr/' + data_name + '/rkde_gak')
assign_tab.append(assign_rkde_gak)
assign_rkde_gau = main.read_assign('../result/ucr/' + data_name + '/rkde_gau')
assign_tab.append(assign_rkde_gau)
assign_kmeans_tga = main.read_assign('../result/ucr/' + data_name + '/kmeans_tga')
assign_tab.append(assign_kmeans_tga)
assign_kmeans_gak = main.read_assign('../result/ucr/' + data_name + '/kmeans_gak')
assign_tab.append(assign_kmeans_gak)
assign_kmeans_gau = main.read_assign('../result/ucr/' + data_name + '/kmeans_gau')
assign_tab.append(assign_kmeans_gau)
assign_dtw_dba = main.read_assign('../result/ucr/' + data_name + '/dtw_dba')
assign_tab.append(assign_dtw_dba)
kappas = np.zeros((n_tests, n_tests))
for i in range(n_tests):
    for j in range(n_tests):
        kappas[i, j] = kappa.kappa_perm(assign_tab[i], assign_tab[j])[0]
print('done\n')

# Stockage des resultats (marshal)
file = open('../result/ucr/' + data_name + '/kappas', 'wb')
marshal.dump(kappas, file)
file.close()

# Stockage des resultats (tableau latex)
f = open('../result/ucr/' + data_name + '/kappas_tex', 'w')
f.write('\\begin{tabular}{c|c}\n')
f.write('method & kappa \\\\\n\hline\n')
f.write('RKDE (tga) & ' + str(kappas[0, 1]) + '\\\\\n')
f.write('RKDE (gak) & ' + str(kappas[0, 2]) + '\\\\\n')
f.write('RKDE (gau) & ' + str(kappas[0, 3]) + '\\\\\n')
f.write('$k$-means (tga) & ' + str(kappas[0, 4]) + '\\\\\n')
f.write('$k$-means (gak) & ' + str(kappas[0, 5]) + '\\\\\n')
f.write('$k$-means (gau) & ' + str(kappas[0, 6]) + '\\\\\n')
f.write('DTW/DBA & ' + str(kappas[0, 7]) + '\\\\\n')
f.write('\\end{tabular}')
f.close()

n_rows = len(kappas)

columns = ('original', 'rkde_tga', 'rkde_gak', 'rkde_gau', 'kmeans_tga', 'kmeans_gak', 'kmeans_gau', 'dtw/dba')
rows = ('original', 'rkde_tga', 'rkde_gak', 'rkde_gau', 'kmeans_tga', 'kmeans_gak', 'kmeans_gau', 'dtw/dba')

cell_text = []
for row in range(n_rows):
    cell_text.append([str(kappas[row][i]) for i in range(n_rows)])

kappas_table = plt.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='center')
plt.show()

# ameliorer T
# sigma fixe (formule Cutturi) + var cutoff pour recherche param'
# Visualiser les resultats (cartes clustering ?)

"""

# Visualisation des resultats

data_name = 'Lighting2'

print('Loading data...')
data, assign_original = load_time_series_ucr(data_name)
assign_rkde_tga = main.read_assign('../result/ucr/' + data_name + '/rkde_tga')
assign_rkde_gak = main.read_assign('../result/ucr/' + data_name + '/rkde_gak')
assign_rkde_gau = main.read_assign('../result/ucr/' + data_name + '/rkde_gau')
assign_kmeans_tga = main.read_assign('../result/ucr/' + data_name + '/kmeans_tga')
assign_kmeans_gak = main.read_assign('../result/ucr/' + data_name + '/kmeans_gak')
assign_kmeans_gau = main.read_assign('../result/ucr/' + data_name + '/kmeans_gau')
assign_dtw_dba = main.read_assign('../result/ucr/' + data_name + '/dtw_dba')
print('Done')

print('Pre-processing...')
assign_original = remove_outliers(assign_original, 1)
nb_class = max(assign_original) + 1
nbr_data = len(data)
len_data = len(data[0])
print('done\n')

# Visualisation en une seule image
pylab.figure()
colors = "bgrcmk"
ax = np.linspace(0, len_data, len_data)
for i in range(nbr_data):
    pylab.plot(ax, data[i], colors[assign_original[i]] + ":", label='continuous')
pylab.show()


# Visualisation en plusieurs images
"""
# 6 images (synthetic_control)
f, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3, sharex='col', sharey='row')
assign_tmp = assign_kmeans_gau
for i in range(nbr_data):
    if assign_tmp[i] == 0:
        ax0.plot(ax, data[i], colors[assign_tmp[i]] + ":", label='continuous')
    if assign_tmp[i] == 1:
        ax1.plot(ax, data[i], colors[assign_tmp[i]] + ":", label='continuous')
    if assign_tmp[i] == 2:
        ax2.plot(ax, data[i], colors[assign_tmp[i]] + ":", label='continuous')
    if assign_tmp[i] == 3:
        ax3.plot(ax, data[i], colors[assign_tmp[i]] + ":", label='continuous')
    if assign_tmp[i] == 4:
        ax4.plot(ax, data[i], colors[assign_tmp[i]] + ":", label='continuous')
    if assign_tmp[i] == 5:
        ax5.plot(ax, data[i], colors[assign_tmp[i]] + ":", label='continuous')
plt.show()

# 4 images (Trace)
f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharex='col', sharey='row')
assign_tmp = assign_rkde_gau
for i in range(nbr_data):
    if assign_tmp[i] == 0:
        ax0.plot(ax, data[i], colors[assign_tmp[i]] + ":", label='continuous')
    if assign_tmp[i] == 1:
        ax1.plot(ax, data[i], colors[assign_tmp[i]] + ":", label='continuous')
    if assign_tmp[i] == 2:
        ax2.plot(ax, data[i], colors[assign_tmp[i]] + ":", label='continuous')
    if assign_tmp[i] == 3:
        ax3.plot(ax, data[i], colors[assign_tmp[i]] + ":", label='continuous')
plt.show()
"""

# 2 images (Lighting2)
f, ((ax0), (ax1)) = plt.subplots(2, 1, sharex='col', sharey='row')
assign_tmp = assign_rkde_gau
for i in range(nbr_data):
    if assign_tmp[i] == 0:
        ax0.plot(ax, data[i], colors[assign_tmp[i]] + ":", label='continuous')
    if assign_tmp[i] == 1:
        ax1.plot(ax, data[i], colors[assign_tmp[i]] + ":", label='continuous')
plt.show()
