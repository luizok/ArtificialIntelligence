import numpy as np
from random import random


# def mse(w, s, t):
#     err = 0

#     for i in range(len(s)):
#         err += sum((t[i] - np.matmul(w.iloc[0, i], s[i]))**2)

#     print("ERROR = " + str(err))
#     return err/len(s)


class LinearRegression():
    np.set_printoptions(precision=4)
    def __init__(self, learn_rate=10e-10, samples=None, targets=None, iter_max=1000):
        self.targets = targets
        self.learn_rate = learn_rate
        self.iter_max = iter_max
        self.epochs = 0
        self.init_weights = np.array(
            [[random() for _ in range(1 + len(samples[0]))] for _ in range(len(targets[0]))]
        )
        self.end_weights = np.array([])

        _samples = []
        for sample in samples:
            _samples.append(np.insert(sample, 0, 1))

        self.samples = np.array(_samples)

        print("dim(T) = {}".format(self.targets.shape))
        print("dim(X) = {}".format(self.samples.shape))
        print("dim(w) = {}".format(self.init_weights.shape))


    def fit(self):
        last_error = 0
        curr_error = float('inf')
        weights = self.init_weights
        epochs = 0
        # mse = Mean Squared Error
        # w = vector of weights
        # s = self.samples
        # t = self. targets
        error = lambda w, s, t: sum(
            t[i] - np.matmul(w, s[i])
            for i in range(len(s))
        )
        
        mse = lambda w, s, t: sum(
            sum((t[i][0] - np.matmul(w, s[i]))**2)
            for i in range(len(s)) 
        )/len(s)

        while self.epochs < self.iter_max:
            
            self.epochs += 1
            print("Trainning epoch = {:05d}".format(self.epochs), end='\r', flush=True)

            for i, x in enumerate(self.samples):
                #print("x = " + str(x))
                #print("w = " + str(weights))
                wx = np.matmul(weights, np.transpose(x))
                #print("dim(wx) = {}".format(wx.shape))
                #print("wx = " + str(wx[0]))
                #print("Ti = " + str(self.targets[i][0]))
                err = self.targets[i][0] - wx[0]
                #print("err = " + str(err))
                #print("dim(err) = {}".format(err.shape))
                weights = weights + self.learn_rate * err * x
                #print(weights)
                #input()

            curr_error = mse(weights, self.samples, self.targets)
            print(("MSE = {:010.6f}" + 10*" ").format(curr_error))

        self.end_weights = weights
        # print("INIT WEIGHTS = {}".format(self.init_weights))
        # print("END WEIGHTS = {}".format(self.end_weights))

        