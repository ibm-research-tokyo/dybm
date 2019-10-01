# (C) Copyright IBM Corp. 2019
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


"""Run experiments for Figure 1 of Dynamic Determinantal Point Processes, AAAI-18.

"""

__author__ = "Takayuki Osogami, Rudy Raymond"


from DyDPP import DyDPP, DataDyDPP
from copy import deepcopy
import numpy as np
import os
import sys
import pickle
import time
import itertools


def shift_data(dataArray, minItems=1):

    """
    The music data is a list of vectors each of which
    has elements of integer in the range [21, 108].
    We substract elements of the vector with 21 so that
    all elements of the vector lie in [0, 87].
    """

    # data is a list of songs, and each song is a list
    # of time-series of notes.
    # data[0] is song[0] = [notes[0], notes[1], ..., notes[M-1]],
    # where notes[T] is a list of notes at time=T
    data = []
    maxNotes = []
    for each in dataArray:
        # eliminate data with no choice or a single choice
        # by "if len(each[i]) > 1"
        data.append([np.array(each[i]) - 21 for i in range(len(each))
                     if len(each[i]) >= minItems])
        maxNotes.append(np.max([len(x) for x in data[-1]]))
    return data, max(maxNotes)


def filter_data(train, valid, test):

    """
    remove notes of test not in train or valid, and renumber the notes
    """

    set88 = set(range(88))
    allKnownSet = set()
    for song in train:
        for notes in song:
            if len(notes) > 0:
                allKnownSet.update(notes)
    for song in valid:
        for notes in song:
            if len(notes) > 0:
                allKnownSet.update(notes)

    # mapping known notes into consecutive number
    mapKnown = dict()
    for each in sorted(allKnownSet):
        mapKnown[each] = len(mapKnown)

    unKnownNotes = set88 - allKnownSet
    if len(unKnownNotes) <= 0:
        return train, valid, test, len(mapKnown)
    else:
        # remove unknownNotes from test
        for i, song in enumerate(test):
            for j, note in enumerate(song):
                test[i][j] = [x for x in note if x not in unKnownNotes]

        # renumber train, valid, and test
        for i, song in enumerate(train):
            for j, note in enumerate(song):
                train[i][j] = [mapKnown[x] for x in note]
        for i, song in enumerate(valid):
            for j, note in enumerate(song):
                valid[i][j] = [mapKnown[x] for x in note]
        for i, song in enumerate(test):
            for j, note in enumerate(song):
                test[i][j] = [mapKnown[x] for x in note]
        return train, valid, test, len(mapKnown)


def ts2DyDPPData(dataArray, nChoices=88, D=1):
    """
    return a list of DyPPData
    """
    answer = []
    M = nChoices
    for each in dataArray:
        T = len(each)
        dyppData = DataDyDPP(T, D, M)
        for t, vec in enumerate(each):
            dyppData.set_selection(t, set(vec))
        answer.append(dyppData)
    return answer


def perform_experiment(train, valid, test, learnRates=[1.0],
                       rank_factors=[1, 2, 3], lags=[1, 2, 3],
                       minEpoch=5, nEpoch=100, L1=0., L2s=[0.],
                       nChoices=88, maxData=None, isRegulazingB=False,
                       alphas=[0.], ReduceAll=False):
    """
    Train with predefined hyperparameters and return the one
    with the best validation log likelihood
    """
    # store the training and validation log likelihood
    # key: (nEpoch, D, K, learningRate, L1, L2),
    # values = (BtrainLL, trainLL, validLL)
    llDict = dict()

    M = nChoices
    static_B = dict()

    for D in lags:
        dydppTrainData = ts2DyDPPData(train, M, D)
        dydppValidData = ts2DyDPPData(valid, M, D)
        dydppTestData = ts2DyDPPData(test, M, D)

        if maxData is not None:
            MAXDATA = min((len(dydppTrainData), maxData))
            dydppTrainData = dydppTrainData[: MAXDATA]
            if ReduceAll:
                dydppValidData = dydppValidData[: MAXDATA]
                dydppTestData = dydppTestData[: MAXDATA]

        maxTrain = max([data.get_max_subset_size() for data in dydppTrainData])
        maxValid = max([data.get_max_subset_size() for data in dydppValidData])
        max_subset_size = max([maxTrain, maxValid])

        ranks = max_subset_size * np.array(rank_factors)

        for L2, K, alpha in itertools.product(L2s, ranks, alphas):
            K = int(K)

            if K < max_subset_size:
                print("Skipping rank=%d smaller than max_subset_size=%d"
                      % (K, max_subset_size))
                continue

            print("=== (D,K,L2,alpha):", D, K, L2, alpha, "===")

            dydpp = DyDPP(M, K, D, L2=L2)
            key = (D, K, L2, alpha)

            if key in static_B:
                dydpp.set_B(static_B[key])
            else:
                llDict[key] = {"DPP": {}, "DyDPP": {}}

                # TRAINING B (STATIC)
                trainLL = dydpp.initialize_B_backtrack(dydppTrainData,
                                                       isResetB=False,
                                                       minIteration=minEpoch,
                                                       maxIteration=nEpoch,
                                                       isRegulazingB=isRegulazingB,
                                                       alpha=alpha)
                # checking training/test LL
                validLL = dydpp.get_average_LL(dydppValidData)
                testLL = dydpp.get_average_LL(dydppTestData)
                llDict[key]["DPP"]["trainLL"] = trainLL
                llDict[key]["DPP"]["validLL"] = validLL
                llDict[key]["DPP"]["testLL"] = testLL
                llDict[key]["DPP"]["best model"] = deepcopy(dydpp)
                print("[TRAINING_B_ONLY]: LL=%.5f" % trainLL)
                print("[VALIDATION_B_ONLY]: LL=%.5f" % validLL)
                print("[TEST_B_ONLY]: LL=%.5f" % testLL)
                static_B[key] = deepcopy(dydpp.variables["B"])

            # TRAINING B AND W (DYNAMIC)
            if D >= 1:
                # otherwise, the same as Static DPP
                trainLL = dydpp.learn_dataset_backtrack(dydppTrainData,
                                                        isResetB=False,
                                                        minIteration=minEpoch,
                                                        maxIteration=nEpoch,
                                                        isRegulazingB=isRegulazingB,
                                                        alpha=alpha)

                validLL = dydpp.get_average_LL(dydppValidData)
                testLL = dydpp.get_average_LL(dydppTestData)
            llDict[key]["DyDPP"]["trainLL"] = trainLL
            llDict[key]["DyDPP"]["validLL"] = validLL
            llDict[key]["DyDPP"]["testLL"] = testLL
            llDict[key]["DyDPP"]["best model"] = deepcopy(dydpp)
            print("[TRAINING_B_W]: LL=%.5f" % trainLL)
            print("[VALIDATION_B_W]: LL=%.5f" % validLL)
            print("[TEST_B_W]: LL=%.5f" % testLL)

    # find best key (with the max validation LL)
    bestKey = {"DPP": None, "DyDPP": None}
    bestVal = {"DPP": float("-inf"), "DyDPP": float("-inf")}
    for key in llDict:
        for model in llDict[key]:
            if llDict[key][model]["validLL"] > bestVal[model]:
                bestVal[model] = llDict[key][model]["validLL"]
                bestKey[model] = key
    for model in ["DPP", "DyDPP"]:
        print("BEST CONFIG for", model, ":", bestKey[model],
              " Best Validation LL=", llDict[bestKey[model]][model]["validLL"],
              " Test LL=", llDict[bestKey[model]][model]["testLL"])

    testLL = llDict[bestKey["DyDPP"]]["DyDPP"]["testLL"]

    return bestKey["DyDPP"], llDict[bestKey["DyDPP"]]["DyDPP"], testLL, llDict


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:"+sys.argv[0]+" pickle_data_file")
        sys.exit(1)
    infilename = sys.argv[1]

    with open(infilename, "rb") as f:
        dataset = pickle.load(f)
    train, maxT = shift_data(dataset["train"], minItems=0)
    valid, maxV = shift_data(dataset["valid"], minItems=0)
    test, maxTe = shift_data(dataset["test"], minItems=0)
    train, valid, test, nChoices = filter_data(train, valid, test)

    print("max subset size", maxT, maxV, maxTe)
    print("training length", len(train))

    if "JSB" in infilename:
        rank_factors = [2, 3, 4]
    else:
        rank_factors = [1, 2, 3]
    bestKey, trainValidLL, testLL, bDict \
        = perform_experiment(train, valid, test,
                             nChoices=nChoices,
                             rank_factors=rank_factors)

    logdir = "log/"
    logname = infilename[len(logdir):len(logdir)+4]
    logname += time.strftime("%Y%m%d-%H%M%S", time.localtime())
    logname += ".pickle"
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    with open(logdir+logname, "wb") as f:
        pickle.dump(bDict, f)

    print("===================================")
    for k in bDict:
        print(k, bDict[k])
    print("===================================")

    print("nEpoch, D, K, learningRate, L1, L2", bestKey)
    print("\tTrainLL=", bDict[bestKey]["DyDPP"]["trainLL"],
          ", ValidLL=", bDict[bestKey]["DyDPP"]["validLL"],
          ", TestLL=", bDict[bestKey]["DyDPP"]["testLL"])
