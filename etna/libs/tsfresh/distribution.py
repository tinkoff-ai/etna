"""
MIT LICENCE

Copyright (c) 2016 Maximilian Christ, Blue Yonder GmbH

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import warnings
# Note: Originally copied from tsfresh package (https://github.com/blue-yonder/tsfresh/blob/ff69073bbb4df787fcbf277a611c6b40632e767d/tsfresh/utilities/distribution.py)
def initialize_warnings_in_workers(show_warnings):
    """
    Small helper function to initialize warnings module in multiprocessing workers.

    On Windows, Python spawns fresh processes which do not inherit from warnings
    state, so warnings must be enabled/disabled before running computations.

    :param show_warnings: whether to show warnings or not.
    :type show_warnings: bool
    """
    warnings.catch_warnings()
    if not show_warnings:
        warnings.simplefilter("ignore")
    else:
        warnings.simplefilter("default")
