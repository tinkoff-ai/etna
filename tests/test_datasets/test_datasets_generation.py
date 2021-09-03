import numpy as np

from etna.datasets.datasets_generation import generate_ar_df
from etna.datasets.datasets_generation import generate_const_df
from etna.datasets.datasets_generation import generate_from_patterns_df
from etna.datasets.datasets_generation import generate_periodic_df


def test_simple_ar_process_check():
    ar_coef = [10, 11]
    random_seed = 1
    random_numbers = np.random.RandomState(seed=random_seed).normal(size=(2, 10))
    ar_df = generate_ar_df(periods=10, start_time="2020-01-01", n_segments=2, ar_coef=ar_coef, random_seed=random_seed)

    assert len(ar_df) == 2 * 10
    assert ar_df.iat[0, 2] == random_numbers[0, 0]
    assert ar_df.iat[1, 2] == ar_coef[0] * ar_df.iat[0, 2] + random_numbers[0, 1]
    assert ar_df.iat[2, 2] == ar_coef[1] * ar_df.iat[0, 2] + ar_coef[0] * ar_df.iat[1, 2] + random_numbers[0, 2]


def test_simple_periodic_df_check():
    period = 3
    periodic_df = generate_periodic_df(periods=11, start_time="2020-01-01", n_segments=2, period=period)
    assert len(periodic_df) == 2 * 11
    assert periodic_df.iat[0, 2] == periodic_df.iat[0 + period, 2]
    assert periodic_df.iat[1, 2] == periodic_df.iat[1 + period, 2]
    assert periodic_df.iat[3, 2] == periodic_df.iat[3 + period, 2]


def test_simple_const_df_check():
    const = 1
    const_df = generate_const_df(start_time="2020-01-01", n_segments=2, periods=3, scale=const)
    assert len(const_df) == 2 * 3
    assert const_df.iat[0, 2] == const
    assert const_df.iat[1, 2] == const
    assert const_df.iat[3, 2] == const


def test_simple_from_patterns_df_check():
    patterns = [[0, 1], [0, 2, 1]]
    from_patterns_df = generate_from_patterns_df(start_time="2020-01-01", patterns=patterns, periods=10)
    assert len(from_patterns_df) == len(patterns) * 10
    assert from_patterns_df[from_patterns_df.segment == "segment_0"].iat[0, 2] == patterns[0][0]
    assert from_patterns_df[from_patterns_df.segment == "segment_0"].iat[1, 2] == patterns[0][1]
    assert from_patterns_df[from_patterns_df.segment == "segment_0"].iat[2, 2] == patterns[0][0]
    assert from_patterns_df[from_patterns_df.segment == "segment_1"].iat[0, 2] == patterns[1][0]
    assert from_patterns_df[from_patterns_df.segment == "segment_1"].iat[3, 2] == patterns[1][0]
    assert from_patterns_df[from_patterns_df.segment == "segment_1"].iat[4, 2] == patterns[1][1]
