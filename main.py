import numpy as np
import pandas as pd
import NeuralNetworks.NeuralNetworks as n_networks
import Regression.Regression as regressor


def parse_csv_file(filename, features_index, targets_index):
    data_set = pd.read_csv(filename)

    features = pd.DataFrame()
    targets = pd.DataFrame()

    for i, name in enumerate(data_set.columns):
        if i in targets_index:
            targets[name] = pd.Series(data_set.iloc[:, i])

        if i in features_index:
            features[name] = pd.Series(data_set.iloc[:, i])

    return features.values, targets.values


if __name__ == "__main__":

    targets = (9, 10)
    samples, targets = parse_csv_file('trainning_sets/data_house.csv',
        (i for i in range(14) if i not in targets), targets
    )
    # inputs, expected = parse_csv_file('data_and_predicts/1_predict.csv', 3)

    r = regressor.LinearRegression(samples=samples, targets=targets, iter_max=10**5)
    r.fit()

    # ann = n_networks.Adaline(learn_rate=.001, samples=samples, targets=targets)
    # ann.train()
    
    # print('\nPREDICT <iter> : <shot> = <expected> ? <True | False>')
    # for i, v in enumerate(inputs):
    #     print('PREDICT {:5d} : {:2d} = {:2d} ? {}'
    #             .format(
    #                 i+1, ann.predict(v),
    #                 expected[i].item(0),
    #                 ann.predict(v) == expected[i].item(0)
    #             )
    #     )