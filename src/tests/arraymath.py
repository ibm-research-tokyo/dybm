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
