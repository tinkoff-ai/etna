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
# Note: Copied from tsfresh package (https://github.com/blue-yonder/tsfresh/blob/ff69073bbb4df787fcbf277a611c6b40632e767d/tsfresh/defaults.py)
CHUNKSIZE = None
N_PROCESSES = 1
PROFILING = False
PROFILING_SORTING = "cumulative"
PROFILING_FILENAME = "profile.txt"
IMPUTE_FUNCTION = None
DISABLE_PROGRESSBAR = False
SHOW_WARNINGS = False
PARALLELISATION = True
TEST_FOR_BINARY_TARGET_BINARY_FEATURE = "fisher"
TEST_FOR_BINARY_TARGET_REAL_FEATURE = "mann"
TEST_FOR_REAL_TARGET_BINARY_FEATURE = "mann"
TEST_FOR_REAL_TARGET_REAL_FEATURE = "kendall"
FDR_LEVEL = 0.05
HYPOTHESES_INDEPENDENT = False
WRITE_SELECTION_REPORT = False
RESULT_DIR = "logging"
