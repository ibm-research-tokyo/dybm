import pydybm.arraymath as amath
import pydybm.arraymath.dynumpy as dynumpy


class NumpyTestMixin(object):
    def setUp(self):
        amath.setup(dynumpy)
        print('\nnumpy test')
        setup = getattr(super(NumpyTestMixin, self), 'setUp', None)
        if setup is not None:
            setup()


class CupyTestMixin(object):
    def setUp(self):
        try:
            import pydybm.arraymath.dycupy as dycupy
            amath.setup(dycupy)
            print('\ncupy test')
            setup = getattr(super(CupyTestMixin, self), 'setUp', None)
            if setup is not None:
                setup()
        except ImportError:
            print('cupy test skipped')
            self.skipTest(
                'cupy is not installed and tests with cupy are passed')
