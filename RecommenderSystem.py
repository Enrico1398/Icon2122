import pandas as pd
import numpy as np
import Preprocess as pp
from matplotlib import pyplot as plt
from kmodes.kmodes import KModes
from fuzzywuzzy import fuzz

import sys

# calcolo della similarità tra userInputGame e ogni medoide trovato
def similarities(data_medoids, userInputGame):
    data_medoids['sum'] = 0

    for i in range(0, len(data_medoids)):
        rowSum = fuzz.ratio(data_medoids['genre'].values[i], userInputGame['genre'].values[0])
        rowSum = rowSum + fuzz.ratio(data_medoids['title'].values[i], userInputGame['title'].values[0])
        rowSum = rowSum + fuzz.ratio(data_medoids['year_range'].values[i], userInputGame['year_range'].values[0])
        rowSum = rowSum + fuzz.ratio(data_medoids['publisher'].values[i], userInputGame['publisher'].values[0])
        rowSum = rowSum + fuzz.ratio(data_medoids['characteristic'].values[i],
                                     userInputGame['characteristic'].values[0])
        rowSum = rowSum + fuzz.ratio(data_medoids['platform'].values[i], userInputGame['platform'].values[0])
        rowSum = rowSum + fuzz.ratio(data_medoids['user_avg_range'].values[i], userInputGame['user_avg'].values[0])
        data_medoids['sum'].values[i] = rowSum

# discretizzazione ed eliminazione di alcune colonne
def dataOperations(dataset):

    bins = [3, 4, 5, 6, 7, 8, 9, 9.5, 10]
    names = ['<4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-9.5', '9.5>']
    bins2 = [90, 91, 92, 93, 94, 95, 96, 97, 98, np.inf]
    names2 = ['<91', '91-92', '92-93', '93-94', '94-95', '95-96', '96-97', '97-98', '98>']

    dataset['metascore_range'] = pd.cut(dataset['metascore'], bins2, labels=names2)
    dataset = dataset.drop(['metascore'], axis=1)
    dataset = dataset.dropna(subset=['metascore_range'])
    dataset['user_avg_range'] = pd.cut(dataset['user_avg'], bins, labels=names)
    dataset = dataset.drop(['user_avg'], axis=1)
    dataset = dataset.dropna(subset=['user_avg_range'])
    dataset = dataset.drop(columns=['no_players'])
    dataset = dataset.drop(columns=['metascore_range'])
    return dataset

# metodo del gomito
def elbowMethod(dataset):
    cost = []
    K = range(1, 10)
    for num_clusters in list(K):
        km = KModes(n_clusters=num_clusters, init="random", n_init=5, verbose=1)
        km.fit_predict(dataset)
        cost.append(km.cost_)

    plt.plot(K, cost, 'bx-')
    plt.xlabel('No. of clusters')
    plt.ylabel('Cost')
    plt.title('Elbow Method For Optimal k')
    plt.show()


'''Definizione di dataframe Pandas separati per ciascun cluster, rimuovendo 
    la colonna indicatrice del numero di cluster corrispondente per ciascuna row.'''

# eliminazione della colonna cluster
def clusterOperations(dataset, n):
    cluster = dataset[dataset.cluster == n]
    cluster = cluster.drop(columns=['cluster'])
    return cluster

# associazione del cluster ai medoidi
def clusterAssociation(data_medoids):
    data_medoids["cluster"] = 0
    for i in range(0, len(data_medoids)):
        data_medoids["cluster"].values[i] = i
        i = i + 1

# calcolo similarità con gli elementi del cluster
def similaritesCluster(cluster, userInputGame):
    cluster['sum'] = 0
    for i in range(0, len(cluster)):
        rowSum = fuzz.ratio(cluster['genre'].values[i], userInputGame['genre'].values[0])
        rowSum = rowSum + fuzz.ratio(cluster['title'].values[i], userInputGame['title'].values[0])
        rowSum = rowSum + fuzz.ratio(cluster['year_range'].values[i], userInputGame['year_range'].values[0])
        rowSum = rowSum + fuzz.ratio(cluster['publisher'].values[i], userInputGame['publisher'].values[0])
        rowSum = rowSum + fuzz.ratio(cluster['characteristic'].values[i], userInputGame['characteristic'].values[0])
        rowSum = rowSum + fuzz.ratio(cluster['platform'].values[i], userInputGame['platform'].values[0])
        rowSum = rowSum + fuzz.ratio(cluster['user_avg_range'].values[i], userInputGame['user_avg'].values[0])
        cluster['sum'].values[i] = rowSum

# scelta del cluster in base al miglior medoide trovato
def clusterSelection(userInputGame, new_data_medoids, cluster1, cluster2, cluster3, cluster4, cluster5, cluster6):
    if new_data_medoids['cluster'].values[0] == 0:
        cluster_1 = pd.DataFrame(cluster1,
                                    columns=['title', 'year_range', 'publisher', 'genre', 'characteristic', 'platform',
                                             'user_avg_range'])
        cluster_1.to_csv(r'..\Icon22\cluster1.csv', index=False)
        cluster_1 = pd.read_csv(r'..\Icon22\cluster1.csv', sep=',')
        similaritesCluster(cluster_1,userInputGame)
        cluster_1.sort_values(by=['sum'], ascending=False, inplace=True)
        input("I videogiochi raccomandati in base alle tue preferenze sono:\n")
        for i in range(0, 5):
            input(cluster_1['title'].values[i])
    elif new_data_medoids['cluster'].values[0] == 1:
        cluster_2 = pd.DataFrame(cluster2,
                                 columns=['title', 'year_range', 'publisher', 'genre', 'characteristic', 'platform',
                                          'user_avg_range'])
        cluster_2.to_csv(r'..\Icon22\cluster2.csv', index=False)
        cluster_2 = pd.read_csv(r'..\Icon22\cluster2.csv', sep=',')
        similaritesCluster(cluster_2,userInputGame)
        cluster_2.sort_values(by=['sum'], ascending=False, inplace=True)
        input("I videogiochi raccomandati in base alle tue preferenze sono:\n")
        for i in range(0, 5):
            input(cluster_2['title'].values[i])
    elif new_data_medoids['cluster'].values[0] == 2:
        cluster_3 = pd.DataFrame(cluster3,
                                 columns=['title', 'year_range', 'publisher', 'genre', 'characteristic', 'platform',
                                          'user_avg_range'])
        cluster_3.to_csv(r'..\Icon22\cluster3.csv', index=False)
        cluster_3 = pd.read_csv(r'..\Icon22\cluster3.csv', sep=',')
        similaritesCluster(cluster_3,userInputGame)
        cluster_3.sort_values(by=['sum'], ascending=False, inplace=True)
        input("I videogiochi raccomandati in base alle tue preferenze sono:\n")
        for i in range(0, 5):
            input(cluster_3['title'].values[i])
    elif new_data_medoids['cluster'].values[0] == 3:
        cluster_4 = pd.DataFrame(cluster4,
                                 columns=['title', 'year_range', 'publisher', 'genre', 'characteristic', 'platform',
                                          'user_avg_range'])
        cluster_4.to_csv(r'..\Icon22\cluster4.csv', index=False)
        cluster_4 = pd.read_csv(r'..\Icon22\cluster4.csv', sep=',')
        similaritesCluster(cluster_4,userInputGame)
        cluster_4.sort_values(by=['sum'], ascending=False, inplace=True)
        input("I videogiochi raccomandati in base alle tue preferenze sono:\n")
        for i in range(0, 5):
            input(cluster_4['title'].values[i])
    elif new_data_medoids['cluster'].values[0] == 4:
        cluster_5 = pd.DataFrame(cluster5,
                                 columns=['title', 'year_range', 'publisher', 'genre', 'characteristic', 'platform',
                                          'user_avg_range'])
        cluster_5.to_csv(r'..\Icon22\cluster5.csv', index=False)
        cluster_5 = pd.read_csv(r'..\Icon22\cluster5.csv', sep=',')
        similaritesCluster(cluster_5,userInputGame)
        cluster_5.sort_values(by=['sum'], ascending=False, inplace=True)
        input("I videogiochi raccomandati in base alle tue preferenze sono:\n")
        for i in range(0, 5):
            input(cluster_5['title'].values[i])
    elif new_data_medoids['cluster'].values[0] == '5':
        cluster_6 = pd.DataFrame(cluster6,
                                 columns=['title', 'year_range', 'publisher', 'genre', 'characteristic', 'platform',
                                          'user_avg_range'])
        cluster_6.to_csv(r'..\Icon22\cluster6.csv', index=False)
        cluster_6 = pd.read_csv(r'..\Icon22\cluster6.csv', sep=',')
        similaritesCluster(cluster_6,userInputGame)
        cluster_6.sort_values(by=['sum'], ascending=False, inplace=True)
        input("I videogiochi raccomandati in base alle tue preferenze sono:\n")
        for i in range(0, 5):
            input(cluster_6['title'].values[i])


def main(userInputGame):
    dataset = pp.main()

    dataset = dataOperations(dataset)
    #elbowMethod(dataset)


    kmode = KModes(n_clusters=6, init="random", n_init=5, verbose=1)
    dataset['cluster'] = kmode.fit_predict(dataset)

    medoids = kmode.cluster_centroids_

    cluster1 = clusterOperations(dataset, 0)
    cluster2 = clusterOperations(dataset, 1)
    cluster3 = clusterOperations(dataset, 2)
    cluster4 = clusterOperations(dataset, 3)
    cluster5 = clusterOperations(dataset, 4)
    cluster6 = clusterOperations(dataset, 5)

    data_medoids = pd.DataFrame(medoids,columns=['title', 'year_range', 'publisher', 'genre', 'characteristic', 'platform','user_avg_range'])
    data_medoids.to_csv(r'..\Icon22\medoid_dataset.csv', index=False)
    data_medoids = pd.read_csv(r'..\Icon22\medoid_dataset.csv', sep=',')
    similarities(data_medoids, userInputGame)

    clusterAssociation(data_medoids)
    data_medoids.sort_values(["sum"], axis=0, ascending=False, inplace=True, na_position='first')

    new_data_medoids = pd.DataFrame(data_medoids,columns=['title', 'year_range', 'publisher', 'genre', 'characteristic', 'platform','user_avg_range', 'sum', 'cluster'])
    new_data_medoids.to_csv(r'..\Icon22\new_data_medoid_dataset.csv', index=False)
    new_data_medoids = pd.read_csv(r'..\Icon22\new_data_medoid_dataset.csv', sep=',')
    clusterSelection(userInputGame, new_data_medoids, cluster1, cluster2, cluster3, cluster4, cluster5, cluster6)


if __name__ == "__main__":
    main(sys.argv[1])
