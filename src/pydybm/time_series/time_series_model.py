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


from abc import ABCMeta, abstractmethod
from six.moves import zip
from .. import arraymath as amath
from ..base.generator import Generator


class TimeSeriesModel(Generator):

    """
    Abstract time series model
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def init_state(self):
        """
        initializing internal state
        """
        pass

    @abstractmethod
    def _update_state(self, in_pattern):
        """
        updating internal state

        Parameters
        ----------
        in_pattern : array, or list of arrays
            pattern used to update the state
        """
        pass

    @abstractmethod
    def _get_delta(self, pattern, expected=None):
        """
        getting how much we change parameters by learning a given pattern

        Parameters
        ----------
        pattern : array, or list of arrays
            given pattern
        expected : array, or list of arrays, optional
            expected pattern
        """
        pass

    @abstractmethod
    def _update_parameters(self, delta):
        """
        update the parameters by delta

        Parameters
        ----------
        delta : dict, or list of dicts
            amount by which the parameters are updated
        """
        pass

    def learn_one_step(self, out_pattern):
        """
        learning a pattern and updating parameters

        Parameters
        ----------
        out_pattern : array, or list of arrays
            pattern whose log likelihood is to be increased
        """
        delta = self._get_delta(out_pattern)
        if delta is not None:
            self._update_parameters(delta)

    def _learn_sequence(self, in_seq, out_seq=None):
        """
        Learning a function mapping input_seq[:t-1] to output_seq[t]

        Parameters
        ----------
        in_seq : list of lists
            input sequence
        out_seq : list of lists
            output sequence
        """

        if out_seq is None:
            out_seq = in_seq

        for i, in_pattern in enumerate(in_seq):
            out_pattern = out_seq[i]
            self.learn_one_step(out_pattern)
            self._update_state(in_pattern)

    def learn(self, in_generator, out_generator=None, get_result=True):
        """
        Learning a function mapping in_generator[:t-1] to out_generator[t]

        Parameters
        ----------
        in_generator : iterator
            input generator
        out_generator : iterator, optional
            target generator
        get_result : boolean, optional
            whether predictions and actuals are yielded

        Returns
        ----------
        dict
            dictionary of
                "prediction": list of array, shape (out_dim,)
                "actual": list of array, shape (out_dim,)
        """
        predictions = list()
        actuals = list()
        for in_pattern in in_generator:
            if out_generator is None:
                out_pattern = in_pattern
            else:
                out_pattern = out_generator.next()
            if get_result:
                prediction = self.predict_next()
                predictions.append(prediction)
                actuals.append(out_pattern)
            self.learn_one_step(out_pattern)
            self._update_state(in_pattern)
        return {"prediction": predictions, "actual": actuals}

    @abstractmethod
    def predict_next(self):
        """
        Predicting next pattern in a deterministic manner

        Returns
        -------
        prediction : array
            predicted pattern
        """
        pass

    def get_predictions(self, sequence):
        """
        Predicting patterns in a determinstic manner

        Parameters
        ----------
        sequence : list or generator
            input sequence, list or generator of in_patterns

        Returns
        -------
        predictions : list
            list of predictions corresponding to the input sequence
        """
        predictions = list()
        for in_pattern in sequence:
            pred = self.predict_next()
            predictions.append(pred)
            self._update_state(in_pattern)
        return predictions


class StochasticTimeSeriesModel(TimeSeriesModel):

    """
    Abstract stochastic time series model
    """

    @abstractmethod
    def get_LL(self, out_pattern):
        """
        Getting log likelihood of given pattern

        Parameters
        ----------
        out_pattern : array, or list of arrays
            pattern whose log likelihood is calculated
        """
        pass

    def get_LL_sequence(self, in_seq, out_seq=None):
        """
        Getting log likelihoods of patterns in a sequence

        Parameters
        ----------
        in_seq : list
            input sequence for updating states
        out_seq : list, optional
            output sequence for which LL is computed (if None, out_seq=in_seq)

        Returns
        -------
        list
            list of log likelihoods
        """

        if out_seq is None:
            out_seq = in_seq

        LL = list()
        for in_pattern, out_pattern in zip(in_seq, out_seq):
            loglikelihood = self.get_LL(out_pattern)
            LL.append(loglikelihood)
            self._update_state(in_pattern)

        return amath.array(LL, dtype=float)
