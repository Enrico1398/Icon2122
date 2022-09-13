import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Preprocess as pp


from sklearn import model_selection
from sklearn.model_selection import train_test_split


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_validate

# preprocess pre-classificazione
def prepDataset(dataset):

    dataset.drop('metascore',axis=1, inplace=True)
    dataset.drop('no_players', axis=1, inplace=True)

    # genre
    def creazioneArrayGenre(row, array):
        if row['genre'] is not None:
            array.append(row['genre']);

    genre = []
    dataset.apply(lambda row: creazioneArrayGenre(row, genre), axis=1)
    nGenre = len(genre)
    genreDict = {}
    j = 0
    for k in range(nGenre):
        genreDict[genre[k]] = k;
        j = k
    genreDict['unknown'] = j + 1

    def subGenre(row, dizionario):
        if row['genre'] is not None:
            element = row['genre']
            if element in dizionario:
                row['genre'] = genreDict[element]
        return row['genre']

    dataset['genre'] = dataset.apply(lambda row: subGenre(row, genreDict), axis=1)

    # title
    def creazioneArrayTitle(row, array):
        if row['title'] is not None:
            array.append(row['title']);

    title = []
    dataset.apply(lambda row: creazioneArrayTitle(row, title), axis=1)
    nTitle = len(title)
    titleDict = {}
    j = 0
    for i in range(nTitle):
        titleDict[title[i]] = i;
        j = i
    titleDict['unknown'] = j + 1

    def subTitle(row, dizionario):
        if row['title'] is not None:
            element = row['title']
            if element in dizionario:
                row['title'] = titleDict[element]
        return row['title']

    dataset['title'] = dataset.apply(lambda row: subTitle(row, titleDict), axis=1)

    # year range
    def creazioneArrayYear(row, array):
        if row['year_range'] is not None:
            array.append(row['year_range']);

    year = []
    dataset.apply(lambda row: creazioneArrayYear(row, year), axis=1)
    nYear = len(year)
    yearDict = {}
    j = 0
    for k in range(nYear):
        yearDict[year[k]] = k;
        j = k
    yearDict['unknown'] = j + 1

    def subYear(row, dizionario):
        if row['year_range'] is not None:
            element = row['year_range']
            if element in dizionario:
                row['year_range'] = yearDict[element]
        return row['year_range']

    dataset['year_range'] = dataset.apply(lambda row: subYear(row, yearDict), axis=1)

    def creazioneArrayPublisher(row, array):
        if row['publisher'] is not None:
            array.append(row['publisher']);

    publisher = []
    dataset.apply(lambda row: creazioneArrayPublisher(row, publisher), axis=1)
    nPublisher = len(publisher)
    publisherDict = {}
    j = 0
    for i in range(nPublisher):
        publisherDict[publisher[i]] = i;
        j = i
    publisherDict['unknown'] = j + 1

    def subPublisher(row, dizionario):
        if row['publisher'] is not None:
            element = row['publisher']
            if element in dizionario:
                row['publisher'] = publisherDict[element]
        return row['publisher']

    dataset['publisher'] = dataset.apply(lambda row: subPublisher(row, publisherDict), axis=1)
    # characteristic
    def creazioneArrayCharacteristic(row, array):
        if row['characteristic'] is not None:
            array.append(row['characteristic']);

    characteristic = []
    dataset.apply(lambda row: creazioneArrayCharacteristic(row, characteristic), axis=1)
    nCharacteristic = len(characteristic)
    characteristicDict = {}
    j = 0
    for k in range(nCharacteristic):
        characteristicDict[characteristic[k]] = k;
        j = k
    characteristicDict['unknown'] = j + 1

    def subCharacteristic(row, dizionario):
        if row['characteristic'] is not None:
            element = row['characteristic']
            if element in dizionario:
                row['characteristic'] = characteristicDict[element]
        return row['characteristic']

    dataset['characteristic'] = dataset.apply(lambda row: subCharacteristic(row, characteristicDict), axis=1)

    # platform
    def creazioneArrayPlatform(row, array):
        if row['platform'] is not None:
            array.append(row['platform']);

    platform = []
    dataset.apply(lambda row: creazioneArrayPlatform(row, platform), axis=1)
    nPlatform = len(platform)
    platformDict = {}
    j = 0
    for p in range(nPlatform):
        platformDict[platform[p]] = p;
        j = p
    platformDict['unknown'] = j + 1

    def subPlatform(row, dizionario):
        if row['platform'] is not None:
            element = row['platform']
            if element in dizionario.keys():
                row['platform'] = platformDict[element]
        return row['platform']

    dataset['platform'] = dataset.apply(lambda row: subPlatform(row, platformDict), axis=1)

    # user_avg
    def creazioneArrayUser_avg(row, array):
        if row['user_avg'] is not None:
            array.append(row['user_avg']);

    user_avg = []
    dataset.apply(lambda row: creazioneArrayUser_avg(row, user_avg), axis=1)
    nUser_avg = len(user_avg)
    user_avgDict = {}
    j = 0
    for p in range(nUser_avg):
        user_avgDict[user_avg[p]] = p;
        j = p
    user_avgDict['unknown'] = j + 1

    def subUser_avg(row, dizionario):
        if row['user_avg'] is not None:
            element = row['user_avg']
            if element in dizionario.keys():
                row['user_avg'] = user_avgDict[element]
        return row['user_avg']

    dataset['user_avg'] = dataset.apply(lambda row: subUser_avg(row, user_avgDict), axis=1)

    # costruzione dataset definendo colonna target
    y = dataset['genre']  # colonna target
    dataset.drop('genre', axis=1, inplace=True)
    x = dataset  # training set

    # bilanciamento
    ros = RandomOverSampler(sampling_strategy="not majority")
    X_res, y_res = ros.fit_resample(x, y)

    # split
    xtr, xts, ytr, yts = train_test_split(X_res, y_res, test_size=0.3, random_state=0)

    class prepElements:
        x_train = xtr
        y_train = ytr
        x_test = xts
        y_test = yts
        genreD = genreDict
        titleD = titleDict
        yearD = yearDict
        publisherD = publisherDict
        characteristicD = characteristicDict
        platformD = platformDict
        user_avgD = user_avgDict
        X_train_complete = X_res
        y_train_complete = y_res

    prep = prepElements()
    return (prep)

# Comparazione Algoritmi
def models_comparison(X, y):
    # preparazione configuratione per cross validation test harness
    # preparazione modelli
    Kfold = model_selection.KFold(n_splits=10, random_state=None)

    svc_model = SVC(C=1.0, gamma='auto', kernel='rbf', probability=True)
    rand_model = RandomForestClassifier(max_features='sqrt', n_estimators=100)
    tree_model = tree.DecisionTreeClassifier()


    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='macro', zero_division=0),
               'recall': make_scorer(recall_score, average='macro', zero_division=0)}

    #  cross-validation su ogni classifier

    svc = cross_validate(svc_model, X, y, cv=Kfold, scoring=scoring)

    rfc = cross_validate(rand_model, X, y, cv=Kfold, scoring=scoring)

    tree1 = cross_validate(tree_model, X, y, cv= Kfold, scoring=scoring)

    # crea un dataframe con i valori delle metriche
    models_scores_table = pd.DataFrame({'SVC': [svc['test_accuracy'].mean(),
                                                svc['test_precision'].mean(),
                                                svc['test_recall'].mean()],

                                        'Random Forest': [rfc['test_accuracy'].mean(),
                                                          rfc['test_precision'].mean(),
                                                          rfc['test_recall'].mean()],

                                        'Decision Tree' : [tree1['test_accuracy'].mean(),
                                                          tree1['test_precision'].mean(),
                                                          tree1['test_recall'].mean()]},
                                        index=['Accuracy', 'Precision', 'Recall'])

    acc = [round(svc['test_accuracy'].mean(), 2),round(rfc['test_accuracy'].mean(), 2),round(tree1['test_accuracy'].mean(), 2)]
    prec = [round(svc['test_precision'].mean(), 2),round(rfc['test_precision'].mean(), 2),round(tree1['test_precision'].mean(), 2)]
    rec = [round(svc['test_recall'].mean(), 2),round(rfc['test_recall'].mean(), 2),round(tree1['test_recall'].mean(), 2)]

    # Add 'Best Score' column
    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
    # tabella dei risultati
    print(models_scores_table)

    # restituisce i risultati delle metriche
    return (models_scores_table, prec, rec, acc)


# plotting dei risultati del confronto in un grafico
def plotResults(prec, rec, acc):
    # plot dei risultati della valutazione
    labels = ['SVC', 'RandFor','DecisionTree']

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, prec, width, label='Precision', align='edge')
    rects2 = ax.bar(x, rec, width, label='Recall', align='edge')
    rects3 = ax.bar(x + width, acc, width, label='Accuracy', align='edge')

    ax.set_ylabel('Scores')
    ax.set_title('Scores by metric and classificator')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='center')

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    fig.tight_layout()

    plt.show()

def finalClassification(X,y):

    model = tree.DecisionTreeClassifier()
    model.fit(X, y)
    filename = 'DecisionTree.sav'
    joblib.dump(model, filename)


# Funzione per la predizione dell'attributo target 'genre'
def predictionGenre(filename,userInput, prepElement):
    user = userInput.copy()

    user.drop('genre', axis=1, inplace=True)

    if user.at[0, 'title'] in prepElement.titleD.keys():
        user.at[0, 'title'] = prepElement.titleD[user.at[0, 'title']]
    else:
        user.at[0, 'title'] = prepElement.titleD['unknown']

    if user.at[0, 'year_range'] in prepElement.yearD.keys():
        user.at[0, 'year_range'] = prepElement.yearD[user.at[0, 'year_range']]
    else:
        user.at[0, 'year_range'] = prepElement.yearD['unknown']

    if user.at[0, 'publisher'] in prepElement.publisherD.keys():
        user.at[0, 'publisher'] = prepElement.publisherD[user.at[0, 'publisher']]
    else:
        user.at[0, 'publisher'] = prepElement.publisherD['unknown']

    if user.at[0, 'characteristic'] in prepElement.characteristicD.keys():
        user.at[0, 'characteristic'] = prepElement.characteristicD[user.at[0, 'characteristic']]
    else:
        user.at[0, 'characteristic'] = prepElement.characteristicD['unknown']

    if user.at[0, 'platform'] in prepElement.platformD.keys():
        user.at[0, 'platform'] = prepElement.platformD[user.at[0, 'platform']]
    else:
        user.at[0, 'platform'] = prepElement.platformD['unknown']

    if user.at[0, 'user_avg'] in prepElement.user_avgD.keys():
        user.at[0, 'user_avg'] = prepElement.user_avgD[user.at[0, 'platform']]
    else:
        user.at[0, 'user_avg'] = prepElement.user_avgD['unknown']



    model = joblib.load(filename)
    gen = model.predict(user)

    for genre, val in prepElement.genreD.items():
        if val == gen:
            gen = genre

    userInput.at[0, 'genre'] = gen
    return gen


def main(userInputGame):

    dataset = pp.main()

    prepInfo = prepDataset(dataset)



    #model, prec, rec, acc = models_comparison(prepInfo.X_train_complete,prepInfo.y_train_complete)#verifica se devi usare xtr,ytr

    # plotting dei risultati in un grafico
    #plotResults(prec,rec,acc)

    #finalClassification(prepInfo.X_train_complete, prepInfo.y_train_complete)

    # predizione del genere del file dato dall'utente
    result = predictionGenre(r"DecisionTree.sav",userInputGame, prepInfo)
    print('Il genere del videogioco da te inserito è %s \n' % result)



if __name__ == "__main__":
    main(sys.argv[1])
