import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from kmodes.kmodes import KModes

import sys


def similarities(medoid, userInputGame):
    totalSum = 0
    medoid['sum'] = 0
    for i in range(0, len(medoid)):
        rowSum = 0
        totalSum = rowSum + totalSum
    return totalSum


def dataOperations(dataset):
    # discretizzazione colonna ratings
    bins = [0, 5, np.inf]
    names = ['<5', '>5']


    dataset['metascore_range'] = pd.cut(dataset['metascore'], bins, labels=names)
    dataset = dataset.drop(['metascore'], axis=1)
    dataset = dataset.dropna(subset=['metascore_range'])
    dataset['user_avg_range'] = pd.cut(dataset['user_avg'], bins, labels=names)
    dataset = dataset.drop(['user_avg'], axis=1)
    dataset = dataset.dropna(subset=['user_avg_range'])
    dataset = dataset.drop(columns=['id'])
    dataset = dataset.drop(columns=['position'])
    return dataset


def elbowMethod(dataset):
    cost = []
    K = range(1, 20)
    for num_clusters in list(K):
        km = KModes(n_clusters=num_clusters, init="random", n_init=5, verbose=1)
        km.fit_predict(dataset)
        cost.append(km.cost_)
        print(km.cluster_centroids_)
    plt.plot(K, cost, 'bx-')
    plt.xlabel('No. of clusters')
    plt.ylabel('Cost')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    return num_clusters


'''Definizione di dataframe Pandas separati per ciascun cluster, rimuovendo 
    la colonna indicatrice del numero di cluster corrispondente per ciascuna row.'''


def clusterOperations(dataset, n):
    cluster = dataset[dataset.cluster == n]
    cluster = cluster.drop(columns=['cluster'])
    return cluster


def toUser(topTen):
    print('Ti consigliamo di guardare:\n')
    for element in topTen:
        print(element)


'''Definizione dei film da suggerire all utente, mediante calcoli relativi
    le similarità con i cluster.'''


def recommendation(cluster1, cluster2, cluster3, userInputGame):
    totSum1 = similarities(cluster1, userInputGame)
    totSum2 = similarities(cluster2, userInputGame)
    totSum3 = similarities(cluster3, userInputGame)
    simil = [totSum1, totSum2, totSum3]
    choice = simil.index(max(simil))
    if choice == 0:
        cluster1.sort_values(by=['sum'], ascending=False, inplace=True)
        topTen = cluster1['title'].head()
    elif choice == 1:
        cluster2.sort_values(by=['sum'], ascending=False, inplace=True)
        topTen = cluster2['title'].head()
    elif choice == 2:
        cluster3.sort_values(by=['sum'], ascending=False, inplace=True)
        topTen = cluster3['title'].head()
    toUser(topTen)


def main():
    dataset = pd.read_csv(r"C:/Users/sinis/PycharmProjects/Icon22/csv_result-DatasetModificato.csv", sep='\t', error_bad_lines=False)
    dataset.get('metascore')
    num_clusters = elbowMethod(dataset)
    print(num_clusters)

    # Building the model with 3 clusters

    km = KModes(n_clusters=3, init="random", n_init=5, verbose=1)
    dataset['cluster'] = km.fit_predict(dataset)


if __name__ == "__main__":
    main(sys.argv[1])
