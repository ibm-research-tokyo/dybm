# (C) Copyright IBM Corp. 2016
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""Example code to benchmark/compare different DyBM models and
LSTM, SimpleRNN models

Here we use the Keras library for use of standard LSTM and RNN
implementations

Problem setting: weekly stock price prediction problem with time step
regression

Note: the same setting can be easily extended any other time series
modeling/prediction setup

Note: For use of Keras update to latest version Keras 2.0 (pip install keras --upgrade)

Note: In order to run with CPU backend, do CUDA_VISIBLE_DEVICES="" python CompareLSTMandDyBM.py
"""

__author__ = "Sakyasingha Dasgupta <SDASGUP@jp.ibm.com>"
__version__ = "1.0"
__date__ = "24th Sep 2016"

import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
import os

from pydybm.time_series.dybm import LinearDyBM
from pydybm.time_series.rnn_gaussian_dybm import RNNGaussianDyBM, GaussianDyBM
from pydybm.base.sgd import RMSProp
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import mean_squared_error

dir_path = os.path.dirname(os.path.realpath(__file__))
head = os.path.split(dir_path)[0]
parent = os.path.dirname(head)
# load the dataset
dataframe = pandas.read_csv(parent + '/data/daily-total-sunspot-number.csv',
                            usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')


def MSE(y_true, y_pred):
    """
    Mean squared error of a sequence of predicted vectors

    y_true : array, shape(L, N)
    y_pred : array, shape(L, N)

    mean of (dy_1^2 + ... + dy_N^2 ) over L pairs of vectors
    (y_true[i], y_pred[i])
    """
    MSE_each_coordinate = mean_squared_error(y_true, y_pred,
                                             multioutput="raw_values")
    return np.sum(MSE_each_coordinate)


def RMSE(y_true, y_pred):
    """
    Root mean squared error of a sequence of predicted vectors

    y_true : array, shape(L, N)
    y_pred : array, shape(L, N)

    squared root of the mean of (dy_1^2 + ... + dy_N^2 ) over L pairs of
    vectors (y_true[i], y_pred[i])
    """
    return np.sqrt(MSE(y_true, y_pred))


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)


def learn_KerasRNN(trainX, trainY, testX, testY, modelType):
    if modelType == "LSTM":
        print ('********************')
        print ('Learning with LSTM')
        print ('********************')
        saveResults = False
        # create and fit the LSTM network
        hidden_no = 10
        max_epochs = 5
        model = Sequential()
        model.add(LSTM(hidden_no, activation='tanh', input_shape=(None, 1)))
        model.add(Dense(input_dim=hidden_no, units=1,
                        activation='linear'))
        model.compile(loss='mse', optimizer='rmsprop')
        start_time = time.time()
        model.fit(trainX, trainY, epochs=max_epochs, shuffle=False,
                  batch_size=1, verbose=2)
        end_time = time.time() - start_time
        # Estimate model training performance
        trainScore = model.evaluate(trainX, trainY, verbose=0)
        trainScore = math.sqrt(trainScore)

        # trainScore = scaler.inverse_transform(numpy.array([[trainScore]]))
        print('Mean Train Score LSTM: %.5f RMSE' % (trainScore))

        # Estimate model test performance
        testScore = model.evaluate(testX, testY, verbose=0)
        testScore = math.sqrt(testScore)

        # testScore = scaler.inverse_transform(numpy.array([[testScore]]))
        print('Mean Test Score LSTM: %.5f RMSE' % (testScore))
        print ('Per epoch time to learn: %.5f sec.' % (end_time))

        if saveResults:
            filename = "RMSE_LSTM_" + str(hidden_no) + "no_delay" + "_epc" \
                       + str(max_epochs) + ".xml"
            out_file = open(filename, "wb")
            out_file.write('Decay' + '\t' + 'LSTM Train RMSE' + '\t'
                           + 'LSTM TestRMSE' + '\n')

            print ('Currently writing to file: %s' % (out_file.name))

            out_file.close()

    elif modelType == "SimpleRNN":
        print ('************************')
        print ('Learning with SimpleRNN')
        print ('************************')
        saveResults = False
        # create and fit the LSTM network
        hidden_no = 10
        max_epochs = 5
        model = Sequential()
        model.add(SimpleRNN(hidden_no,
                            activation='tanh',
                            input_shape=(None, 1)))
        model.add(Dense(input_dim=hidden_no, units=1,
                        activation='linear'))
        model.compile(loss='mse', optimizer='rmsprop')
        start_time = time.time()
        model.fit(trainX, trainY, epochs=max_epochs, shuffle=False,
                  batch_size=1, verbose=2)
        end_time = time.time() - start_time
        # Estimate model training performance
        trainScore = model.evaluate(trainX, trainY, verbose=0)
        trainScore = math.sqrt(trainScore)

        # trainScore = scaler.inverse_transform(numpy.array([[trainScore]]))
        print('Mean Train Score SimpleRNN: %.5f RMSE' % (trainScore))

        # Estimate model test performance
        testScore = model.evaluate(testX, testY, verbose=0)
        testScore = math.sqrt(testScore)

        # testScore = scaler.inverse_transform(numpy.array([[testScore]]))
        print('Mean Test Score SimpleRNN: %.5f RMSE' % (testScore))
        print ('Per epoch time to learn: %.5f sec.' % (end_time))
        if saveResults:
            filename = "RMSE_SimpleRNN_" + str(hidden_no) + "no_delay" \
                       + "_epc" + str(max_epochs) + ".xml"
            out_file = open(filename, "wb")
            out_file.write('Decay' + '\t' + 'SimpleRNN Train RMSE' + '\t'
                           + 'SimpleRNN TestRMSE' + '\n')

            print ('Currently writing to file: %s' % (out_file.name))

            out_file.close()

    # generate predictions for training
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    plotFig = False

    if(plotFig):
        plotData(trainPredict, testPredict, dataset)


def plotData(trainPredict, testPredict, dataset):

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] \
        = testPredict

    # plot baseline and predictions for a fixed dimension
    plt.title("Time-series prediction")
    # choose the dimension of data to plot
    plt.xlabel('time', fontsize=20)
    plt.ylabel('data', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.plot(dataset, color='r', label='baseline')
    plt.plot(trainPredictPlot, color='b', label='train prediction')
    plt.plot(testPredictPlot, color='k', label='test prediction')
    plt.legend(fontsize=15)
    plt.show()


def learn_DyBM(trainX, trainY, testX, testY, DyBMmodel):

    plotFig = False
    RNN_dim = 10
    input_dim = 1
    max_epochs = 5
    saveResults = False
    SGD = RMSProp

    """
    Choose the DyBM model type: VAR (default), Gaussian, RNNGaussian

    Change accordinly for other DyBM models
    """
    print ('*****************************************')
    print ('Learning with DyBM model %s' % (DyBMmodel))
    print ('*****************************************')

    trainX = np.reshape(trainX, (trainX.shape[0], input_dim))
    testX = np.reshape(testX, (testX.shape[0], input_dim))

    # default config of decay change accordinly
    decay = [0.5]

    for delay in [3]:

        print "learning with delay =", delay

        if saveResults:
            filename = "RMSE_DyBM_" + \
                str(RNN_dim) + "delay_" + str(delay) + ".xml"
            out_file = open(filename, "wb")
            out_file.write('Decay' + '\t' + 'DyBM Train RMSE' + '\t'
                           + 'DyBMTestRMSE' + '\n')

            print ('Currently writing to file: %s' % (out_file.name))

        if DyBMmodel == "VAR":
            dybm = LinearDyBM(input_dim, delay=delay, decay_rates=[0.0])
            dybm.set_learning_rate(0.001)
            dybm.init_state

        elif DyBMmodel == "RNNGaussian":

            dybm = RNNGaussianDyBM(input_dim, input_dim, RNN_dim, 0.3,
                                   0.1, delay, decay_rates=decay, leak=1.0,
                                   SGD=SGD())

            dybm.set_learning_rate(0.001)
            dybm.init_state

        elif DyBMmodel == "Gaussian":
            dybm = GaussianDyBM(input_dim, delay=delay, decay_rates=decay)
            dybm.set_learning_rate(0.001)
            dybm.init_state

        errs = list()
        train_errs = list()
        for epochs in range(max_epochs):

            # training
            start_time = time.time()
            dybm.init_state()
            result = dybm.learn(trainX, get_result=True)
            end_time = time.time() - start_time
            train_rmse = RMSE(trainY, result["prediction"])
            print ('train error: %.5f' % (train_rmse))

            # testing
            # dybm.init_state()
            result2 = dybm.learn(testX, get_result=True)
            test_rmse = RMSE(testY, result2["prediction"])

            print ('test error: %.5f ' % (test_rmse))

            errs.append(test_rmse)
            train_errs.append(train_rmse)

            error = np.array(errs)

            curr_mean = np.mean(error)

            print ('delay = %f decay = %s curr_mean_error = %f' % (delay, decay, curr_mean))

        print ('Mean train error with DyBM: %.5f RMSE'
               % (np.mean(np.array(train_errs))))
        print ('Mean test error with DyBM: %.5f RMSE' % (curr_mean))
        print ('Per epoch time to learn: %.5f sec.' % (end_time))

    if saveResults:
        out_file.write('%s\t%s\t%s\n'
                       % (str(decay), str(train_rmse), str(test_rmse)))

        out_file.write('\n%s\t%s'
                       % ('Mean test error with DyBM', str(curr_mean)))
        out_file.close()


if __name__ == "__main__":

    np.random.seed(2)
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    print "Number of Observations/datapoints in each dimension", len(dataset)
    # split into train and test sets (default 60% train, 40% test)

    trainPercentage = 0.6
    train_size = int(len(dataset) * trainPercentage)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)

    # Check training data size
    print "Train data X shape: ", trainX.shape
    print "Train data Y shape: ", trainY.shape

    testX, testY = create_dataset(test, look_back)

    # Check test data size
    print "Test data X shape: ", testX.shape
    print "Test data Y shape: ", testY.shape

    learn_DyBM(trainX, trainY, testX, testY, DyBMmodel="RNNGaussian")

    learn_KerasRNN(trainX, trainY, testX, testY, modelType="LSTM")
    # learn_KerasRNN(trainX, trainY, testX, testY, modelType="SimpleRNN")


