from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from numpy.random import RandomState
from statsmodels.tsa.arima_process import arma_generate_sample


def generate_ar_df(
    periods: int,
    start_time: str,
    ar_coef: Optional[list] = None,
    sigma: float = 1,
    n_segments: int = 1,
    freq: str = "1D",
    random_seed: int = 1,
) -> pd.DataFrame:
    """
    Create DataFrame with AR process data.

    Parameters
    ----------
    periods:
        number of timestamps
    start_time:
        start timestamp
    ar_coef:
        AR coefficients
    sigma:
        scale of AR noise
    n_segments:
        number of segments
    freq:
        pandas frequency string for :py:func:`pandas.date_range` that is used to generate timestamp
    random_seed:
        random seed
    """
    if ar_coef is None:
        ar_coef = [1]
    random_sampler = RandomState(seed=random_seed).normal
    ar_coef = np.r_[1, -np.array(ar_coef)]
    ar_samples = arma_generate_sample(
        ar=ar_coef, ma=[1], nsample=(n_segments, periods), axis=1, distrvs=random_sampler, scale=sigma
    )
    df = pd.DataFrame(data=ar_samples.T, columns=[f"segment_{i}" for i in range(n_segments)])
    df["timestamp"] = pd.date_range(start=start_time, freq=freq, periods=periods)
    df = df.melt(id_vars=["timestamp"], value_name="target", var_name="segment")
    return df


def generate_periodic_df(
    periods: int,
    start_time: str,
    scale: float = 10,
    period: int = 1,
    n_segments: int = 1,
    freq: str = "1D",
    add_noise: bool = False,
    sigma: float = 1,
    random_seed: int = 1,
) -> pd.DataFrame:
    """
    Create DataFrame with periodic data.

    Parameters
    ----------
    periods:
        number of timestamps
    start_time:
        start timestamp
    scale:
        we sample data from Uniform[0, scale)
    period:
        data frequency -- x[i+period] = x[i]
    n_segments:
        number of segments
    freq:
        pandas frequency string for :py:func:`pandas.date_range` that is used to generate timestamp
    add_noise:
        if True we add noise to final samples
    sigma:
        scale of added noise
    random_seed:
        random seed
    """
    samples = RandomState(seed=random_seed).randint(int(scale), size=(n_segments, period))
    patterns = [list(ar) for ar in samples]
    df = generate_from_patterns_df(
        periods=periods,
        start_time=start_time,
        patterns=patterns,
        sigma=sigma,
        random_seed=random_seed,
        freq=freq,
        add_noise=add_noise,
    )
    return df


def generate_const_df(
    periods: int,
    start_time: str,
    scale: float,
    n_segments: int = 1,
    freq: str = "1D",
    add_noise: bool = False,
    sigma: float = 1,
    random_seed: int = 1,
) -> pd.DataFrame:
    """
    Create DataFrame with const data.

    Parameters
    ----------
    periods:
        number of timestamps
    start_time:
        start timestamp
    scale:
        const value to fill
    period:
        data frequency -- x[i+period] = x[i]
    n_segments:
        number of segments
    freq:
        pandas frequency string for :py:func:`pandas.date_range` that is used to generate timestamp
    add_noise:
        if True we add noise to final samples
    sigma:
        scale of added noise
    random_seed:
        random seed
    """
    patterns = [[scale] for _ in range(n_segments)]
    df = generate_from_patterns_df(
        periods=periods,
        start_time=start_time,
        patterns=patterns,
        sigma=sigma,
        random_seed=random_seed,
        freq=freq,
        add_noise=add_noise,
    )
    return df


def generate_from_patterns_df(
    periods: int,
    start_time: str,
    patterns: List[List[float]],
    freq: str = "1D",
    add_noise=False,
    sigma: float = 1,
    random_seed: int = 1,
) -> pd.DataFrame:
    """
    Create DataFrame from patterns.

    Parameters
    ----------
    periods:
        number of timestamps
    start_time:
        start timestamp
    patterns:
        list of lists with patterns to be repeated
    freq:
        pandas frequency string for :py:func:`pandas.date_range` that is used to generate timestamp
    add_noise:
        if True we add noise to final samples
    sigma:
        scale of added noise
    random_seed:
        random seed
    """
    n_segments = len(patterns)
    if add_noise:
        noise = RandomState(seed=random_seed).normal(scale=sigma, size=(n_segments, periods))
    else:
        noise = np.zeros(shape=(n_segments, periods))
    samples = noise
    for idx, pattern in enumerate(patterns):
        samples[idx, :] += np.array(pattern * (periods // len(pattern) + 1))[:periods]
    df = pd.DataFrame(data=samples.T, columns=[f"segment_{i}" for i in range(n_segments)])
    df["timestamp"] = pd.date_range(start=start_time, freq=freq, periods=periods)
    df = df.melt(id_vars=["timestamp"], value_name="target", var_name="segment")
    return df
