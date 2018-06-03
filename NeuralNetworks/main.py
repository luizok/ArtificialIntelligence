import numpy as np
import pandas as pd
import NeuralNetworks as n_networks


def parse_csv_file(filename, n_features):
    data_set = pd.read_csv(filename)

    x = data_set.iloc[:, :n_features].values
    targets = data_set.iloc[:, n_features:].values

    return x, targets


if __name__ == "__main__":

    samples, targets = parse_csv_file('data_and_predicts/1_dataset.csv', 3)
    inputs, expected = parse_csv_file('data_and_predicts/1_predict.csv', 3)

    ann = n_networks.Adaline(learn_rate=.001, samples=samples, targets=targets)
    ann.train()
    
    print('\nPREDICT <iter> : <shot> = <expected> ? <True | False>')
    for i, v in enumerate(inputs):
        print('PREDICT {:5d} : {:2d} = {:2d} ? {}'
                .format(
                    i+1, ann.predict(v),
                    expected[i].item(0),
                    ann.predict(v) == expected[i].item(0)
                )
        )