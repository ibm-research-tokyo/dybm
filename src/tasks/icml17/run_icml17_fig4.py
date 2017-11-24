# (C) Copyright IBM Corp. 2017
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

"""
Script for running experiments for Figure 4 in our ICML 2017 paper.

python run_icml17_fig4.py sunspot test 1000 30 0 0 0.0
python run_icml17_fig4.py sunspot test 1000 30 4 0 0.0
python run_icml17_fig4.py sunspot test 1000 30 4 2 0.25
python run_icml17_fig4.py sunspot test 1000 30 4 2 0.5
python run_icml17_fig4.py sunspot test 1000 30 4 2 1.0
python run_icml17_fig4.py price test 10000 3 0 0 0.0
python run_icml17_fig4.py price test 10000 3 4 0 0.0
python run_icml17_fig4.py price test 10000 3 4 2 0.25
python run_icml17_fig4.py price test 10000 3 4 2 0.5
python run_icml17_fig4.py price test 10000 3 4 2 1.0
python run_icml17_fig4.py climate test 1000 2 0 0 0.0
python run_icml17_fig4.py climate test 1000 2 4 0 0.0
python run_icml17_fig4.py climate test 1000 2 4 2 0.25
python run_icml17_fig4.py climate test 1000 2 4 2 0.5
python run_icml17_fig4.py climate test 1000 2 4 2 1.0

After obtaining results, run plot_icml17_fig4.py to make the figure.

.. seealso:: Takayuki Osogami, Hiroshi Kajino, and Taro Sekiyama, \
"Bidirectional learning for time-series models with hidden units", \
ICML 2017.
"""

__author__ = "Takayuki Osogami"


import pydybm.arraymath as amath
import pydybm.arraymath.dycupy as dycupy
# Uncomment the following line to run with cupy, which however
# slows down execution due to large overhead for instances
# of this size
# amath.setup(dycupy)
import argparse
import os
import numpy as np
import pickle
from pydybm.time_series.dybm import GaussianBernoulliDyBM
from pydybm.base.generator import SequenceGenerator
from pydybm.base.metrics import RMSE


class DatasetGenerator(SequenceGenerator):
    def __init__(self, data, start, end):
        data = data[start:end]
        data = data.tolist()
        SequenceGenerator.__init__(self, data)


class Dataset():
    def __init__(self, npy, len_warm, len_train, len_test):

        dataset = np.load(npy)

        # scale with respect to training data
        train = dataset[:(len_train + 2 * len_warm)]
        scale = train.max(axis=0) - train.min(axis=0)
        dataset = (dataset - train.min(axis=0)) / scale

        t0 = 0
        t1 = t0 + len_warm
        t2 = t1 + len_train
        t3 = t2 + len_warm
        t4 = t3 + len_test
        self.warmup = DatasetGenerator(dataset, t0, t1)
        self.train = DatasetGenerator(dataset, t1, t2)
        self.cooldown = DatasetGenerator(dataset, t2, t3)
        self.test = DatasetGenerator(dataset, t3, t4)

        self.test_seq = self.test.to_list()


def get_dataset(dataset_name, evaluation, len_warm):

    if dataset_name == 'price':
        npy = '../../../data/PET_PRI_GND_A_EPM0_PTE_DPGAL_W.npy'
        len_data = 1223     # 1223 in total
        if evaluation == 'val':
            len_train = len_data // 3
            len_test = len_data - 2 * len_train
            len_train = len_train - 2 * len_warm
        elif evaluation == 'test':
            len_train = 819
            len_test = len_data - len_train
            len_train = len_train - 2 * len_warm
        else:
            print('Unknown eval: {}'.format(evaluation))
            exit(1)
    elif dataset_name == 'sunspot':
        npy = '../../../data/monthly-sunspot-number-zurich-17.npy'
        len_data = 2820
        if evaluation == 'val':
            len_train = len_data // 3
            len_test = len_data - 2 * len_train
            len_train = len_train - 2 * len_warm
        elif evaluation == 'test':
            len_train = 1889
            len_test = len_data - len_train
            len_train = len_train - 2 * len_warm
        else:
            print('Unknown eval: {}'.format(evaluation))
            exit(1)
    elif dataset_name == "climate":
        npy = "../../../data/air.mon.anom.npy"
        len_data = 1635
        if evaluation == "val":
            len_train = len_data * 2 // 5  # 654
            len_test = len_train           # 654
            len_train = len_train - 2 * len_warm
        elif evaluation == 'test':
            len_test = 327
            len_train = 1635 - len_test - 2 * len_warm
        else:
            print('Unknown eval: {}'.format(evaluation))
            exit(1)
    else:
        print('Unknown dataset: {}'.format(dataset_name))
        exit(1)

    dataset = Dataset(npy, len_warm, len_train, len_test)

    return dataset, len_train, len_test


def get_init_rate(dataset, delay, Nh, sigma, epochs):
    Nv = dataset.train.get_dim()        # number of visible units
    decay = [0.0]

    dybm = GaussianBernoulliDyBM([delay, delay], [decay, decay],
                                 [Nv, Nh], sigma=sigma)
    if len(dybm.layers[0].layers[0].variables["W"]) > 0:
        dybm.layers[0].layers[0].variables["W"][0] += amath.eye(Nv)

    dataset.train.reset()
    prediction = dybm.get_predictions(dataset.train)
    init_train_rmse = RMSE(dataset.train.to_list(), prediction)

    print("init train rmse", init_train_rmse)

    best_rate = 1.0
    rate = 1.0
    best_train_rmse = init_train_rmse
    while rate > 0:
        # prepare a dybm
        dybm = GaussianBernoulliDyBM([delay, delay], [decay, decay],
                                     [Nv, Nh], sigma=sigma)

        if len(dybm.layers[0].layers[0].variables["W"]) > 0:
            dybm.layers[0].layers[0].variables["W"][0] += amath.eye(Nv)
        for i in range(2):
            dybm.layers[i].layers[0].SGD.set_learning_rate(rate)
            dybm.layers[i].layers[1].SGD.set_learning_rate(0)

        # train epochs
        for i in range(epochs):
            dataset.warmup.reset()
            dataset.train.reset()
            dybm.get_predictions(dataset.warmup)
            dybm.learn(dataset.train, get_result=False)

        dybm.init_state()
        dataset.warmup.reset()
        dataset.train.reset()
        dybm.get_predictions(dataset.warmup)
        prediction = dybm.get_predictions(dataset.train)
        train_rmse = RMSE(dataset.train.to_list(), prediction)

        print("rate", rate, "train_rmse", train_rmse)

        if train_rmse < best_train_rmse:
            best_train_rmse = train_rmse

        if train_rmse > best_train_rmse and train_rmse < init_train_rmse:
            best_rate = rate * 2
            break

        rate = rate * 0.5

    if best_rate == 1.0:
        rate = 2.0
        while rate < amath.inf:
            # prepare a dybm
            dybm = GaussianBernoulliDyBM([delay, delay], [decay, decay],
                                         [Nv, Nh], sigma=sigma)

            if len(dybm.layers[0].layers[0].variables["W"]) > 0:
                dybm.layers[0].layers[0].variables["W"][0] += amath.eye(Nv)
            for i in range(2):
                dybm.layers[i].layers[0].SGD.set_learning_rate(rate)
                dybm.layers[i].layers[1].SGD.set_learning_rate(0)

            # train one epoch
            dataset.warmup.reset()
            dataset.train.reset()
            dybm.get_predictions(dataset.warmup)
            dybm.learn(dataset.train, get_result=False)

            dybm.init_state()
            dataset.warmup.reset()
            dataset.train.reset()
            dybm.get_predictions(dataset.warmup)
            prediction = dybm.get_predictions(dataset.train)
            train_rmse = RMSE(dataset.train.to_list(), prediction)

            print("rate", rate, "train_rmse", train_rmse)

            if train_rmse > init_train_rmse:
                best_rate = rate / 2.
                rate = rate / 2.
                break

            rate = rate * 2

    print("best initial learning rate", best_rate)

    return best_rate


def experiment(dataset, delay, Nh, repeat, bi_factor, bi_end, sigma, rate):
    """
    A run of experiment

    Parameters
    ----------
    delay : int
        delay
    Nh : int
        number of hidden units
    repeat : int
        number of iterations of training
    bi_factor : boolean
        amount of bidirectional training
    rate : float
        initial learning rate
    sigma : float
        standard deviation of noise
    """

    Nv = dataset.train.get_dim()        # number of visible units
    decay = [0.0]

    # Prepare a Gaussian Bernoulli DyBM

    dybm = GaussianBernoulliDyBM([delay, delay], [decay, decay],
                                 [Nv, Nh], sigma=sigma)

    if len(dybm.layers[0].layers[0].variables["W"]) > 0:
        dybm.layers[0].layers[0].variables["W"][0] += amath.eye(Nv)
    for i in range(2):
        dybm.layers[i].layers[0].SGD.set_learning_rate(rate)
        dybm.layers[i].layers[1].SGD.set_learning_rate(0)

    # Learn

    train_rmse = list()
    test_rmse = list()
    step = list()

    dybm.init_state()
    dataset.warmup.reset()
    dataset.train.reset()
    dybm.get_predictions(dataset.warmup)
    prediction = dybm.get_predictions(dataset.train)
    rmse = RMSE(dataset.train.to_list(), prediction)
    rmse = amath.to_numpy(rmse)
    train_rmse.append(rmse)

    print("init", rmse)

    dybm.init_state()
    dataset.cooldown.reset()
    dataset.test.reset()
    dybm.get_predictions(dataset.cooldown)
    prediction = dybm.get_predictions(dataset.test)
    rmse = RMSE(dataset.test_seq, prediction)
    rmse = amath.to_numpy(rmse)

    print(rmse)

    test_rmse.append(rmse)

    step.append(0)

    for i in range(repeat):

        dybm.init_state()
        dataset.warmup.reset()
        dataset.train.reset()
        dataset.cooldown.reset()

        if i % (bi_factor + 1) == 0 and bi_factor > 0 and i < repeat * bi_end:
            print("backward")
            # make a time-reversed DyBM and dataset
            dybm._time_reversal()
            dataset.warmup.reverse()
            dataset.train.reverse()
            dataset.cooldown.reverse()

            # update internal states by reading backward sequence
            dybm.get_predictions(dataset.cooldown)

            # learn backward sequence
            dybm.learn(dataset.train, get_result=False)
            dybm.learn(dataset.warmup, get_result=False)

            # make a non time-reversed DyBM
            dybm._time_reversal()
            dataset.warmup.reverse()
            dataset.train.reverse()
            dataset.cooldown.reverse()
        else:
            print("forward")
            # update internal states by reading forward sequence
            dybm.get_predictions(dataset.warmup)

            # learn forward sequence
            dybm.learn(dataset.train, get_result=False)
            dybm.learn(dataset.cooldown, get_result=False)

        if i % (bi_factor + 1) == bi_factor:
            print("evaluate")

            dybm.init_state()
            dataset.warmup.reset()
            dataset.train.reset()
            dybm.get_predictions(dataset.warmup)
            prediction = dybm.get_predictions(dataset.train)
            rmse = RMSE(dataset.train.to_list(), prediction)
            rmse = amath.to_numpy(rmse)
            train_rmse.append(rmse)

            print(i, rmse)

            dybm.init_state()
            dataset.cooldown.reset()
            dataset.test.reset()
            dybm.get_predictions(dataset.cooldown)
            prediction = dybm.get_predictions(dataset.test)
            rmse = RMSE(dataset.test_seq, prediction)
            rmse = amath.to_numpy(rmse)
            test_rmse.append(rmse)

            print(rmse)

            step.append(i + 1)

    return(train_rmse, test_rmse, step, dybm)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluation of bidirectional training')
    parser.add_argument('dataset', help='`price` or `sunspot` or `climate`')
    parser.add_argument('eval', help=('`val` (training+validation) or '
                                      '`test` (training+validation+test)'))
    parser.add_argument("repeat", help="repeat: 1000, 10000,...")
    parser.add_argument("delay", help="delay: 4, ...")
    parser.add_argument("Nh", help="Number of hidden units: 0, 4, ...")
    parser.add_argument("bi_factor", help="bi_factor: 2, ...")
    parser.add_argument("bi_end", help="float: 0.0, 0.25, 0.5, 1.0, ...")
    args = parser.parse_args()

    print(args)

    repeat = int(args.repeat)
    delay = int(args.delay)
    Nh = int(args.Nh)
    bi_factor = int(args.bi_factor)
    bi_end = float(args.bi_end)

    if args.dataset in ["climate"]:
        sigma = 0.001
        filename = args.dataset + "_" + args.eval \
                   + "_repeat" + args.repeat \
                   + "_delay" + args.delay \
                   + "_Nh" + args.Nh \
                   + "_bi" + args.bi_factor \
                   + "_end" + args.bi_end \
                   + "_std" + str(sigma)
    else:
        sigma = 0.01
        filename = args.dataset + "_" + args.eval \
                   + "_repeat" + args.repeat \
                   + "_delay" + args.delay \
                   + "_Nh" + args.Nh \
                   + "_bi" + args.bi_factor \
                   + "_end" + args.bi_end

    # Prepare data generators

    len_warm = delay + 1
    dataset, len_train, len_test = get_dataset(args.dataset,
                                               args.eval,
                                               len_warm)

    rate = get_init_rate(dataset, delay, Nh, sigma, repeat / 100)

    print("initial learning rate", rate)

    directory = args.dataset + "_" + args.eval + '_results_largest' \
                + str(repeat / 100) + "/"
    if not os.path.exists(directory):
        print("Creating directory " + directory)
        os.mkdir(directory)

    if os.path.exists(directory + filename + "_error.npy") \
       and os.path.exists(directory + filename + "_error.npy"):
        error = np.load(directory + filename + "_error.npy")
        steps = np.load(directory + filename + "_steps.npy")
    else:
        train_rmse, test_rmse, steps, dybm \
            = experiment(dataset, delay, Nh, repeat, bi_factor,
                         bi_end, sigma, rate)

        print train_rmse

        # dump results
        train_rmse = np.array(train_rmse)
        train_rmse.dump(directory + filename + "_train.npy")

        test_rmse = np.array(test_rmse)
        test_rmse.dump(directory + filename + "_test.npy")

        steps = np.array(steps)
        steps.dump(directory + filename + "_steps.npy")

        with open(directory + filename + ".pkl", "wb") as f:
            pickle.dump(dybm, f)
