import numpy as np
import matplotlib.pyplot as plt
from random import random


def sign(x):
    return 1 if x >= 0 else -1


class Perceptron():

    def __init__(self, learn_rate=.0001, samples=None, targets=None):
        self.learn_rate = learn_rate
        self.samples = None
        self.targets = targets
        self.epochs = 0
        self.init_weights = np.array([random() for _ in range(len(samples[0])+1)])
        self.end_weights = None

        _samples = []
        for sample in samples:
            _samples.append(np.insert(sample, 0, -1))

        self.samples = np.array(_samples)

    
    def train(self, plot=True):
        weights = self.init_weights
        has_error = True

        ## PLOT_CODE_START ##
        if plot:
            plt.xlabel('Epochs')
            plt.ylabel('nº of errors')
            plt.grid(True, linestyle='dashed')
        ## PLOT_CODE_END ##

        print('INITIAL WEIGHTS  = ' + str(weights))
        while has_error:
            error_count = 0
            has_error = False

            # For each sample, take the sign of sum(wi * xi), for i=1,...,dimension
            # If sign is diferent from the expected one, then there's error, so keep on 
            for i, sample in enumerate(self.samples):
                u = np.matmul(weights, sample)
                y = np.sign(u)

                if y != self.targets[i]:
                    has_error = True
                    error_count += 1
                    weights = weights + self.learn_rate * (self.targets[i] - y) * sample   
            
            ## PLOT_CODE_START ##
            if plot:
                plt.plot(self.epochs, error_count, marker='.', color='red')
            ## PLOT_CODE_END ##

            self.epochs += 1
            print('\r' + 15 * ' ' + '\r', end='', flush=True)
            print("EPOCA {:d}".format(self.epochs), end='', flush=True)

        ## PLOT_CODE_START ##
        if plot:
            plt.show()
        ## PLOT_CODE_END ##

        self.end_weights = weights
        print('\nFINAL WEIGHTS    = ' + str(self.end_weights))


    def verify_samples_results(self):

        for i, sample in enumerate(self.samples):
            u = np.matmul(self.end_weights, sample)
            y = sign(u)

            assert y == self.targets[i]

        print("Training Set is OK")

    
    def predict(self, _input):
        _input = np.array(_input)
        _input = np.insert(_input, 0, -1)

        return sign(np.matmul(self.end_weights, _input))


class Adaline():
    def __init__(self, learn_rate=.0001, samples=None, targets=None, tolerance=10e-5):
        self.learn_rate = learn_rate
        self.samples = None
        self.targets = targets
        self.tolerance = tolerance
        self.epochs = 0
        self.init_weights = np.array([random() for _ in range(len(samples[0])+1)])
        self.end_weights = None

        _samples = []
        for sample in samples:
            _samples.append(np.insert(sample, 0, -1))

        self.samples = np.array(_samples)

    
    def train(self, plot=True):
        weights = self.init_weights
        last_error = 0.0
        curr_error = float('inf')

        x_points = None
        y_points = None

        mean_squared_error = lambda w, samples, targets: sum(
            sum((targets[i] - np.matmul(w, samples[i]))**2)
            for i in range(len(samples))
        )

        ## PLOT_CODE_START ##
        if plot:
            x_points = []
            y_points = []
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.grid(True, linestyle='dashed')
        ## PLOT_CODE_END ##

        print('INITIAL WEIGHTS  = ' + str(weights))
        while abs(curr_error - last_error) > self.tolerance:
            
            last_error = mean_squared_error(weights, self.samples, self.targets)

            # For each sample, calculate sum(wi * xi), for i=1,...,dimension
            # Calculate the new MSE (mean squared error), and keep on if the absolute
            # diference between the new MSE and the old MSE is bigger than tolerance
            for i, sample in enumerate(self.samples):
                u = np.matmul(weights, sample)
                weights = weights + self.learn_rate * (self.targets[i] - u) * sample   
            
            curr_error = mean_squared_error(weights, self.samples, self.targets)

            ## PLOT_CODE_START ##
            if plot:
                x_points.append(self.epochs)
                y_points.append(last_error)
            ## PLOT_CODE_END ##

            self.epochs += 1
            print('\r' + 15 * ' ' + '\r', end='', flush=True)
            print("EPOCA {:d}".format(self.epochs), end='', flush=True)

        ## PLOT_CODE_START ##
        if plot:
            plt.plot(x_points, y_points, color='red', linewidth=0.6)
            plt.show()
        ## PLOT_CODE_END ##

        self.end_weights = weights
        print('\nFINAL WEIGHTS    = ' + str(self.end_weights))


    def verify_samples_results(self):

        print("CHECKING TEST SET\n\n")
        for i, sample in enumerate(self.samples):
            u = np.matmul(self.end_weights, sample)
            y = sign(u)

            print("SAMPLE {:5d} : {:2d} = {:2d} ? {}" \
                    .format(i+1, y, self.targets[i].item(0), y == self.targets[i].item(0))
            )

    
    def predict(self, _input):
        _input = np.array(_input)
        _input = np.insert(_input, 0, -1)

        return sign(np.matmul(self.end_weights, _input))

