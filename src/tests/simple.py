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


__author__ = "Takayuki Osogami"


import numpy as np
from six.moves import xrange, zip

import pydybm.arraymath as amath
from pydybm.base.generator import Uniform, NoisySin, ListGenerator, SequenceGenerator
from pydybm.base.metrics import RMSE, MSE


def test_binary_model(model, max_repeat=1000, generator=False):
    """
    minimal test with learning a constant with noise
    """

    in_dim = model.get_input_dimension()
    out_dim = model.get_target_dimension()
    batch = in_dim
    eps = 1e-2

    in_seq = amath.array([[i % in_dim == j % in_dim for j in range(in_dim)]
                          for i in range(batch)], dtype=int)

    if in_dim == out_dim:
        print("Testing generative learning")
        out_seq = in_seq
    else:
        print("Testing discriminative learning")
        out_seq = amath.array([[i % out_dim == j % out_dim
                                for j in range(out_dim)]
                               for i in range(batch)], dtype=int)

    in_gen = SequenceGenerator(in_seq)
    out_gen = SequenceGenerator(out_seq)

    i = 0
    for i in xrange(max_repeat):
        if in_dim == out_dim:
            if generator:
                in_gen.reset()
                model.learn(in_gen)
            else:
                model._learn_sequence(in_seq)
        else:
            if generator:
                in_gen.reset()
                out_gen.reset()
                model.learn(in_gen, out_gen)
            else:
                model._learn_sequence(in_seq, out_seq)

        if i % 1000 == 0:
            if in_dim == out_dim:
                if generator:
                    in_gen.reset()
                    predictions = model.get_predictions(in_gen)
                else:
                    predictions = model.get_predictions(in_seq)
            else:
                if generator:
                    in_gen.reset()
                    predictions = model.get_predictions(in_gen)
                else:
                    predictions = model.get_predictions(in_seq)

            """
            diffs = predictions - out_seq
            SE = [np.dot(diff, diff) for diff in diffs]
            RMSE2 = np.sqrt(np.mean(SE))
            """

            rmse = RMSE(predictions, out_seq)

            print("%d\t%1.3f" % (i, rmse))
            if rmse < eps * out_dim:
                print("Successfully completed in %d iterations with RMSE: %f"
                      % (i + 1, rmse))
                break

    if in_dim == out_dim:
        LL = model.get_LL_sequence(in_seq)
    else:
        LL = model.get_LL_sequence(in_seq, out_seq)
    print("LL: %f" % amath.mean(LL))

    return i + 1


def test_real_model(model, max_repeat=1000, generator=False):
    """
    minimal test with learning a constant with noise
    """

    in_dim = model.get_input_dimension()
    out_dim = model.get_target_dimension()
    batch = 3
    in_mean = 1.0
    out_mean = 2.0
    d = 0.001

    random = amath.random.RandomState(0)
    in_seq = random.uniform(low=in_mean - d, high=in_mean + d,
                            size=(batch, in_dim))
    in_gen = Uniform(length=batch, low=in_mean - d,
                     high=in_mean + d, dim=in_dim)

    if in_dim == out_dim:
        print("Testing generative learning for a real model")
        out_seq = in_seq
    elif out_dim.__class__ is list:
        random = amath.random.RandomState(0)
        out_seq = list()
        for i in xrange(batch):
            patterns = [random.uniform(low=out_mean - d, high=out_mean + d,
                                       size=dim)
                        for dim in out_dim]
            out_seq.append(patterns)
        out_gens = [Uniform(length=batch, low=out_mean - d, high=out_mean + d,
                            dim=dim)
                    for dim in out_dim]
        out_gen = ListGenerator(out_gens)
    else:
        print("Testing discriminative learning for a real model")
        random = amath.random.RandomState(0)
        out_seq = random.uniform(low=out_mean - d, high=out_mean + d,
                                 size=(batch, out_dim))
        out_gen = Uniform(length=batch, low=out_mean - d, high=out_mean + d,
                          dim=out_dim)

    print("Input dimension: %d" % in_dim)
    if out_dim.__class__ is list:
        print("Target dimension: " + str(out_dim))
    else:
        print("Target dimension: %d" % out_dim)

    i = 0
    for i in xrange(max_repeat):
        if in_dim == out_dim:
            if generator:
                # print("Predicting with input generator of length: %d"
                # % in_gen.limit)
                in_gen.reset()
                model.learn(in_gen)
                in_gen.reset()
                predictions = model.get_predictions(in_gen)
            else:
                # print("Predicting with input sequence")
                model._learn_sequence(in_seq)
                predictions = model.get_predictions(in_seq)
        else:
            if generator:
                # print("Predicting with input generator and target generator")
                in_gen.reset()
                out_gen.reset()
                model.learn(in_gen, out_gen)
            else:
                # print("Predicting with input sequence and target sequence")
                model._learn_sequence(in_seq, out_seq)
            predictions = model.get_predictions(in_seq)
        """
        diffs = predictions - out_seq
        SE = [amath.dot(diff, diff) for diff in diffs]
        rmse = amath.sqrt(amath.mean(SE))
        """
        if out_dim.__class__ is list:
            predictions = [amath.concatenate(pred) for pred in predictions]
            rmse = RMSE(predictions,
                        [amath.concatenate(pat) for pat in out_seq])
        else:
            rmse = RMSE(predictions, out_seq)

        if i % 1000 == 0:
            print("%d\t%1.4f" % (i, rmse))
        if rmse < d:
            print("Successfully completed in %d iterations with RMSE: %f"
                  % (i + 1, rmse))
            break

    op = getattr(model, "get_LL_sequence", None)
    if callable(op):
        if in_dim == out_dim:
            LL = model.get_LL_sequence(in_seq)
        else:
            LL = model.get_LL_sequence(in_seq, out_seq)
            if LL is not None:
                print("LL: %f" % amath.mean(LL))

    return i + 1


def test_complex_model(model, max_repeat=100):

    dim = [layer.get_input_dimension() for layer in model.layers]
    activations = model.activations

    random = amath.random.RandomState(0)
    mean = 1.0
    batch = np.prod(dim)
    d = 0.01

    seqs = list()
    for dimension, activation in zip(dim, activations):
        if activation == "linear":
            seq = random.uniform(low=mean - d, high=mean + d,
                                 size=(batch, dimension))
        elif activation == "sigmoid":
            seq = amath.array([[i % dimension == j % dimension
                                for j in range(dimension)]
                               for i in range(batch)], dtype=int)
            seq = amath.array([[i % dimension == j % dimension
                                for j in range(dimension)]
                               for i in range(batch)],
                              dtype=float)
        seqs.append(seq)
    in_seq = list()

    i = 0
    for i in xrange(batch):
        pattern = [s[i] for s in seqs]
        in_seq.append(pattern)

    for i in xrange(max_repeat):
        model._learn_sequence(in_seq)

        predictions = model.get_predictions(in_seq)
        predictions = np.concatenate(predictions).reshape((len(predictions),
                                                           len(predictions[0])))

        rmse = 0
        start = 0
        j = 0
        for j, (dimension, sequence) in enumerate(zip(dim, seqs)):
            sub_prediction = predictions[:, j]
            sub_prediction = amath.concatenate(sub_prediction).reshape(
                (len(sub_prediction), len(sub_prediction[0])))
            start += dimension
            rmse += amath.sqrt(amath.mean((sub_prediction - sequence)**2))

        if i % 1000 == 0:
            print("rmse at %d:\t%1.4f" % (i, rmse))
        if rmse < d * sum(dim):
            print("Successfully completed in %d iterations with rmse: %f"
                  % (i + 1, rmse))
            break

    LL = model.get_LL_sequence(in_seq)
    print("LL: ", LL)

    return i + 1


def test_sin(model, max_repeat=30):

    random = amath.random.RandomState(0)

    dim = model.get_input_dimension()
    # phase = np.zeros(dim)
    # phase = 2 * np.pi * np.arange(dim) / dim
    phase = 2 * amath.pi * random.uniform(size=dim)

    std = 1.
    period = 100
    length = period

    print("Testing with noisy sin wave")
    print(" dimension: %d" % dim)
    print(" period: %d" % period)
    print(" standard deviation: %f" % std)

    wave = NoisySin(length, period, std, dim, phase=phase)

    y_pred = model.get_predictions(wave)
    wave.reset()
    y_true = [w for w in wave]
    init_error = RMSE(y_true, y_pred)
    print(" 0 Error: %f" % init_error)

    for i in xrange(max_repeat):
        wave.reset()
        result = model.learn(wave)
        y_true = result["actual"]
        y_pred = result["prediction"]
        error = RMSE(y_true, y_pred)
        print("%2d Error: %f" % (i + 1, error))

    reduction = 100 * (init_error - error) / init_error
    assert reduction > 0, \
        "Error increased from %f to %f" % (init_error, error)
    print("Error is reduced from %f to %f by %f percents in %d iterations"
          % (init_error, error, reduction, max_repeat))
