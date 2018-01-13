# Fichier permettant de moduler les differentes methodes de clustering
try:
    # Import generaux
    import numpy as np
    import pylab
    import sys
    import platform
    import matplotlib.pyplot as plt
    import re
    
    # Import locaux
    import kmeans
    import rkde
except:
    exit(1)



"""  Clustering """



# Clusterise les donnees avec la methode desiree
# Entree :
# - M : la matrice des distances entre les objets
# - methode : une chaine de caractere donnant le nom de la methode (nom de module)
# - params : une liste des parametres requis pour la methode demandee
#   - kmeans : params = [k, n_iter]
#   - rkde : params = [bandwidth, prob]
# Sortie :
# - assign : un tableau donnant pour chaque entier (objet) son numero de cluster
# - nb_cluster : le nombre de clusters formes
def make_clusters(M, methode, params):
    function = methode + ".do"
    assign, nb_clusters = eval(function)(M, params[0], params[1])
    return assign, nb_clusters



""" Lecture et affichage de donnees """



# Fonction de lecture dans un fichier
# Entree :
# - file_name : une chaine de caracteres donnant le nom du fichier a ouvrir
# - nb_item : nombre de lignes a lire (-1 pour tout lire, defaut a -1)
# Sortie :
# - data : une liste de liste de flottants
def read_data(file_name, nb_item = -1):
    f = open(file_name,'r')
    data = []
    cpt = 0
    for line in f:
        if (0 <= nb_item and nb_item <= cpt):
            break
        line = re.split('\s+', line) # '\s' matches whitespace characters
        line = [float(x) for x in line if x != '']
        data.append(line)
        cpt += 1
    f.close()
    return data

# Fonction d'affichage d'un nuage de points
# Entree :
# - data : un ensemble de points sous la forme d'une matrice de taille n*2
# - assign : un tableau de taille n representant une assignation de [data]
def show(data, assign):
    colors = "bgrcmyk"
    symbols = ".ov18sp*h+xD_"
    nb_clusters = max(assign) + 1
    pylab.figure()
    mini = min( min(data[:][0]), min(data[:][1]) )
    maxi = max( max(data[i][0]), max(data[i][1]) )
    pylab.xlim([mini, maxi])
    pylab.ylim([mini, maxi])
    
    if (nb_clusters < 8):
        for i_k in range(nb_clusters):
            pylab.plot([data[i][0] for i in range(len(data)) if assign[i] == i_k],
               [data[i][1] for i in range(len(data)) if assign[i] == i_k],
               colors[i_k] + ".")
    else:
        for i_k in range(nb_clusters):
            pylab.plot( [data[i][0] for i in range(len(data)) if assign[i] == i_k],
                [data[i][1] for i in range(len(data)) if assign[i] == i_k],
                colors[i_k % 7]) + symbols[int(i_k / 7)]
    pylab.show()



""" Lecture et ecriture d'une assignation """



# Lis un fichier ou est inscrit une assignation.
# Entree :
# - file : adresse et nom du fichier
# Sortie :
# - assign : un vecteur numpy d'entiers 
def read_assign(file_name):
    f = open(file_name,'r')
    assign_tmp = []
    i = 0
    for line in f:
        try:
            assign_tmp.append(int(line))
            i = i + 1
        except ValueError:
            continue
    f.close()
    
    return np.array(assign_tmp)

# Ecris une assignation dans un fichier
# Entree :
# - file_name : adresse et nom d'un fichier
# - assign : l'assignation a ecrire
# - nb_iter : le nombre d'iterations faites par l'algorithme (-1) s'il n'est pas
#   base sur ce principe
# - s : la seed utilisee pour le clustering
def write_cluster(file_name, assign, nb_iter, s): 
    nb_data = len(assign)
    nb_cluster = max(assign) + 1
    f = open(file_name, 'w')
    f.write('nb_cluster = ' + str(nb_cluster) + '\n')
    f.write('nb_iter = '    + str(nb_iter)    + '\n')
    f.write('nb_data = '    + str(nb_data)    + '\n')
    f.write('seed = '       + str(s)          + '\n')
    for i in assign:
        f.write(str(i) + '\n')
    f.close()



""" Fonctions non encore retravaillees """



# Fonction pour enregistrer des images : 
    # data_file = fichier contenant les donnees
    # assign_file = fichier cree a partir du clustering et contenant la table d'assignation
    # file_figure = nom du fichier dans lequel sera enregistre l'image
    # format = nom de l'extention du fichier cree (pdf,svg,png...)
# exemple : save('cercles/cercles.txt', 'cercles_kmeans', 'figure_cercles_kmeans', 'pdf')
def save(data_file, assign_file,file_figure,format):
    data = read_data(data_file)
    assign = read_assign(data,assign_file)
    nombre_clusters = numpy.amax(assign) +1
    
    plt.ioff()
    fig = plt.figure()
    colors = "bgrcmyk"
    symbols = ".ov18sp*h+xD_"
    mini = min( min([data[i][0] for i in range(len(data))]), min([data[i][1] for i in range(len(data))]) )
    maxi = max( max([data[i][0] for i in range(len(data))]), max([data[i][1] for i in range(len(data))]) )
    plt.xlim([mini, maxi])
    plt.ylim([mini, maxi])
    
    if (nombre_clusters < 8):
        for i_k in range(nombre_clusters):
            plt.plot([data[i][0] for i in range(len(data)) if assign[i] == i_k],
               [data[i][1] for i in range(len(data)) if assign[i] == i_k],
               colors[i_k] + ".")
    else:
        if (nombre_clusters < 85):
            for i_k in range(nombre_clusters):
                plt.plot( [data[i][0] for i in range(len(data)) if assign[i] == i_k],
                    [data[i][1] for i in range(len(data)) if assign[i] == i_k],
                    colors[i_k % 7] + symbols[int(i_k / 7)] )
        else: 
            print("too many clusters")
    if (platform.system() == "Windows"):
        plt.savefig('C:/users/alex/documents/Alex/Cours/ENS/M1_Cours/Projet/data/Results/'+file_figure+'.'+format)
    else: 
        plt.savefig('../data/Results/'+file_figure+'.'+format)
    plt.close(fig)

