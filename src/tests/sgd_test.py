import unittest


class TestCaseSGD(unittest.TestCase):

    def test_vSGD(self):
        from collections import defaultdict
        from pydybm.base.sgd import vSGD
        from pydybm.time_series.dybm import LinearDyBM
        from pydybm.base.generator import Uniform

        gen = Uniform(length=1000, low=0, high=1, dim=1)

        dybm = LinearDyBM(in_dim=1, delay=1, decay_rates=[], SGD=vSGD(hessian=defaultdict(lambda: 1)))
        dybm.learn(gen)

        self.assertEqual(float(dybm.variables['b']), 0.4932564368799297)


if __name__ == '__main__':
    unittest.main()
