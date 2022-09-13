import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def main():
    dataset = pd.read_csv(r"C:/Users/sinis/PycharmProjects/Icon22/csv_result-DatasetModificato4.csv", sep=';', error_bad_lines=False)
    dataset = dataset.drop(columns = ['id','position'])

    # discretizzazione valori della colonna 'year'
    bins = [1990, 1995, 2000, 2005, 2010, 2015, 2020, np.inf]
    names = ['<1995', '1995-2000', '2000-2005',
             '2005-2010', '2010-2015', '2015-2020', '>2020']
    dataset['year_range'] = pd.cut(dataset['year'], bins, labels=names)
    dataset = dataset.drop(['year'], axis=1)

    # da categoriche a numeriche per il titolo
    labelEncoderTitle = LabelEncoder()
    dataset['title'] = labelEncoderTitle.fit_transform(dataset['title'])

    # da categoriche a numeriche per year_range
    labelEncoderYear = LabelEncoder()
    dataset['year_range'] = labelEncoderYear.fit_transform(dataset['year_range'])

    # da categoriche a numeriche per il publisher
    labelEncoderPublisher = LabelEncoder()
    dataset['publisher'] = labelEncoderPublisher.fit_transform(dataset['publisher'])

    # da categoriche a numeriche per il genere
    labelEncoder = LabelEncoder()
    dataset['genre'] = labelEncoder.fit_transform(dataset['genre'])

    # da categoriche a numeriche per il characteristic
    labelEncoderCharacteristic = LabelEncoder()
    dataset['characteristic'] = labelEncoderCharacteristic.fit_transform(dataset['characteristic'])

    # da categoriche a numeriche per il platform
    labelEncoderPlatform = LabelEncoder()
    dataset['platform'] = labelEncoderPlatform.fit_transform(dataset['platform'])

    # da catecorighe a numeriche per il no_players
    labelEncoderNo_Players = LabelEncoder()
    dataset['no_players'] = labelEncoderNo_Players.fit_transform(dataset['no_players'])


    columns = ['title','year_range','publisher','genre','characteristic','platform','no_players']
    df = pd.DataFrame(columns = columns)
    df['title'] = labelEncoderTitle.inverse_transform(dataset['title'])
    df['year_range'] = labelEncoderYear.inverse_transform(dataset['year_range'])
    df['publisher'] = labelEncoderPublisher.inverse_transform(dataset['publisher'])
    df['genre'] = labelEncoder.inverse_transform(dataset['genre'])
    df['characteristic'] = labelEncoderCharacteristic.inverse_transform(dataset['characteristic'])
    df['platform'] = labelEncoderPlatform.inverse_transform(dataset['platform'])
    df['metascore'] = dataset['metascore']
    df['user_avg'] = dataset['user_avg']
    df['no_players'] = labelEncoderNo_Players.inverse_transform(dataset['no_players'])
    df.to_csv(r'..\Icon22\new_dataset.csv', index = False)


    return df