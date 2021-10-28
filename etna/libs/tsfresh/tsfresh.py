import warnings
from builtins import str
from functools import partial
from functools import reduce
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

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


def target_binary_feature_binary_test(x, y):
    """
    Calculate the feature significance of a binary feature to a binary target as a p-value.
    Use the two-sided univariate fisher test from :func:`~scipy.stats.fisher_exact` for this.
    :param x: the binary feature vector
    :type x: pandas.Series
    :param y: the binary target vector
    :type y: pandas.Series
    :return: the p-value of the feature significance test. Lower p-values indicate a higher feature significance
    :rtype: float
    :raise: ``ValueError`` if the target or the feature is not binary.
    """
    __check_if_pandas_series(x, y)
    _check_for_nans(x, y)

    # Check for correct value range
    __check_for_binary_feature(x)
    __check_for_binary_target(y)

    # Extract the unique values
    x0, x1 = np.unique(x.values)
    y0, y1 = np.unique(y.values)

    # Calculate contingency table
    n_y1_x0 = np.sum(y[x == x0] == y1)
    n_y0_x0 = len(y[x == x0]) - n_y1_x0
    n_y1_x1 = np.sum(y[x == x1] == y1)
    n_y0_x1 = len(y[x == x1]) - n_y1_x1

    table = np.array([[n_y1_x1, n_y1_x0], [n_y0_x1, n_y0_x0]])

    # Perform the Fisher test
    oddsratio, p_value = stats.fisher_exact(table, alternative="two-sided")

    return p_value


def target_binary_feature_real_test(x, y, test):
    """
    Calculate the feature significance of a real-valued feature to a binary target as a p-value.
    Use either the `Mann-Whitney U` or `Kolmogorov Smirnov` from  :func:`~scipy.stats.mannwhitneyu` or
    :func:`~scipy.stats.ks_2samp` for this.
    :param x: the real-valued feature vector
    :type x: pandas.Series
    :param y: the binary target vector
    :type y: pandas.Series
    :param test: The significance test to be used. Either ``'mann'`` for the Mann-Whitney-U test
                 or ``'smir'`` for the Kolmogorov-Smirnov test
    :type test: str
    :return: the p-value of the feature significance test. Lower p-values indicate a higher feature significance
    :rtype: float
    :raise: ``ValueError`` if the target is not binary.
    """
    __check_if_pandas_series(x, y)
    _check_for_nans(x, y)

    # Check for correct value range
    __check_for_binary_target(y)

    # Extract the unique values
    y0, y1 = np.unique(y.values)

    # Divide feature according to target
    x_y1 = x[y == y1]
    x_y0 = x[y == y0]

    if test == "mann":
        # Perform Mann-Whitney-U test
        U, p_mannwhitu = stats.mannwhitneyu(x_y1, x_y0, use_continuity=True, alternative="two-sided")
        return p_mannwhitu
    elif test == "smir":
        # Perform Kolmogorov-Smirnov test
        KS, p_ks = stats.ks_2samp(x_y1, x_y0)
        return p_ks
    else:
        raise ValueError(
            "Please use a valid entry for test_for_binary_target_real_feature. "
            + "Valid entries are 'mann' and 'smir'."
        )


def target_real_feature_binary_test(x, y):
    """
    Calculate the feature significance of a binary feature to a real-valued target as a p-value.
    Use the `Kolmogorov-Smirnov` test from from :func:`~scipy.stats.ks_2samp` for this.
    :param x: the binary feature vector
    :type x: pandas.Series
    :param y: the real-valued target vector
    :type y: pandas.Series
    :return: the p-value of the feature significance test. Lower p-values indicate a higher feature significance.
    :rtype: float
    :raise: ``ValueError`` if the feature is not binary.
    """
    __check_if_pandas_series(x, y)
    _check_for_nans(x, y)

    # Check for correct value range
    __check_for_binary_feature(x)

    # Extract the unique values
    x0, x1 = np.unique(x.values)

    # Divide target according to feature
    y_x1 = y[x == x1]
    y_x0 = y[x == x0]

    # Perform Kolmogorov-Smirnov test
    KS, p_value = stats.ks_2samp(y_x1, y_x0)

    return p_value


def target_real_feature_real_test(x, y):
    """
    Calculate the feature significance of a real-valued feature to a real-valued target as a p-value.
    Use `Kendall's tau` from :func:`~scipy.stats.kendalltau` for this.
    :param x: the real-valued feature vector
    :type x: pandas.Series
    :param y: the real-valued target vector
    :type y: pandas.Series
    :return: the p-value of the feature significance test. Lower p-values indicate a higher feature significance.
    :rtype: float
    """
    __check_if_pandas_series(x, y)
    _check_for_nans(x, y)

    tau, p_value = stats.kendalltau(x, y, method="asymptotic")
    return p_value


def __check_if_pandas_series(x, y):
    """
    Helper function to check if both x and y are pandas.Series. If not, raises a ``TypeError``.
    :param x: the first object to check.
    :type x: Any
    :param y: the second object to check.
    :type y: Any
    :return: None
    :rtype: None
    :raise: ``TypeError`` if one of the objects is not a pandas.Series.
    """
    if not isinstance(x, pd.Series):
        raise TypeError("x should be a pandas Series")
    if not isinstance(y, pd.Series):
        raise TypeError("y should be a pandas Series")
    if not list(y.index) == list(x.index):
        raise ValueError("X and y need to have the same index!")


def __check_for_binary_target(y):
    """
    Helper function to check if a target column is binary.
    Checks if only the values true and false (or 0 and 1) are present in the values.
    :param y: the values to check for.
    :type y: pandas.Series or numpy.array
    :return: None
    :rtype: None
    :raises: ``ValueError`` if the values are not binary.
    """
    if not set(y) == {0, 1}:
        if len(set(y)) > 2:
            raise ValueError("Target is not binary!")

        warnings.warn(
            "The binary target should have " "values 1 and 0 (or True and False). " "Instead found" + str(set(y)),
            RuntimeWarning,
        )


def __check_for_binary_feature(x):
    """
    Helper function to check if a feature column is binary.
    Checks if only the values true and false (or 0 and 1) are present in the values.
    :param y: the values to check for.
    :type y: pandas.Series or numpy.array
    :return: None
    :rtype: None
    :raises: ``ValueError`` if the values are not binary.
    """
    if not set(x) == {0, 1}:
        if len(set(x)) > 2:
            raise ValueError("[target_binary_feature_binary_test] Feature is not binary!")

        warnings.warn(
            "A binary feature should have only "
            "values 1 and 0 (incl. True and False). "
            "Instead found " + str(set(x)) + " in feature ''" + str(x.name) + "''.",
            RuntimeWarning,
        )


def _check_for_nans(x, y):
    """
    Helper function to check if target or feature contains NaNs.
    :param x: A feature
    :type x: pandas.Series
    :param y: The target
    :type y: pandas.Series
    :raises: `ValueError` if target or feature contains NaNs.
    """
    if np.isnan(x.values).any():
        raise ValueError("Feature {} contains NaN values".format(x.name))
    elif np.isnan(y.values).any():
        raise ValueError("Target contains NaN values")


def calculate_relevance_table(
    X,
    y,
    ml_task="auto",
    multiclass=False,
    n_significant=1,
    n_jobs=N_PROCESSES,
    show_warnings=SHOW_WARNINGS,
    chunksize=CHUNKSIZE,
    test_for_binary_target_binary_feature=TEST_FOR_BINARY_TARGET_BINARY_FEATURE,
    test_for_binary_target_real_feature=TEST_FOR_BINARY_TARGET_REAL_FEATURE,
    test_for_real_target_binary_feature=TEST_FOR_REAL_TARGET_BINARY_FEATURE,
    test_for_real_target_real_feature=TEST_FOR_REAL_TARGET_REAL_FEATURE,
    fdr_level=FDR_LEVEL,
    hypotheses_independent=HYPOTHESES_INDEPENDENT,
):
    """
    Calculate the relevance table for the features contained in feature matrix `X` with respect to target vector `y`.
    The relevance table is calculated for the intended machine learning task `ml_task`.
    To accomplish this for each feature from the input pandas.DataFrame an univariate feature significance test
    is conducted. Those tests generate p values that are then evaluated by the Benjamini Hochberg procedure to
    decide which features to keep and which to delete.
    We are testing
        :math:`H_0` = the Feature is not relevant and should not be added
    against
        :math:`H_1` = the Feature is relevant and should be kept
    or in other words
        :math:`H_0` = Target and Feature are independent / the Feature has no influence on the target
        :math:`H_1` = Target and Feature are associated / dependent
    When the target is binary this becomes
        :math:`H_0 = \\left( F_{\\text{target}=1} = F_{\\text{target}=0} \\right)`
        :math:`H_1 = \\left( F_{\\text{target}=1} \\neq F_{\\text{target}=0} \\right)`
    Where :math:`F` is the distribution of the target.
    In the same way we can state the hypothesis when the feature is binary
        :math:`H_0 =  \\left( T_{\\text{feature}=1} = T_{\\text{feature}=0} \\right)`
        :math:`H_1 = \\left( T_{\\text{feature}=1} \\neq T_{\\text{feature}=0} \\right)`
    Here :math:`T` is the distribution of the target.
    TODO: And for real valued?
    :param X: Feature matrix in the format mentioned before which will be reduced to only the relevant features.
              It can contain both binary or real-valued features at the same time.
    :type X: pandas.DataFrame
    :param y: Target vector which is needed to test which features are relevant. Can be binary or real-valued.
    :type y: pandas.Series or numpy.ndarray
    :param ml_task: The intended machine learning task. Either `'classification'`, `'regression'` or `'auto'`.
                    Defaults to `'auto'`, meaning the intended task is inferred from `y`.
                    If `y` has a boolean, integer or object dtype, the task is assumed to be classification,
                    else regression.
    :type ml_task: str
    :param multiclass: Whether the problem is multiclass classification. This modifies the way in which features
                       are selected. Multiclass requires the features to be statistically significant for
                       predicting n_significant classes.
    :type multiclass: bool
    :param n_significant: The number of classes for which features should be statistically significant predictors
                          to be regarded as 'relevant'
    :type n_significant: int
    :param test_for_binary_target_binary_feature: Which test to be used for binary target, binary feature
                                                  (currently unused)
    :type test_for_binary_target_binary_feature: str
    :param test_for_binary_target_real_feature: Which test to be used for binary target, real feature
    :type test_for_binary_target_real_feature: str
    :param test_for_real_target_binary_feature: Which test to be used for real target, binary feature (currently unused)
    :type test_for_real_target_binary_feature: str
    :param test_for_real_target_real_feature: Which test to be used for real target, real feature (currently unused)
    :type test_for_real_target_real_feature: str
    :param fdr_level: The FDR level that should be respected, this is the theoretical expected percentage of irrelevant
                      features among all created features.
    :type fdr_level: float
    :param hypotheses_independent: Can the significance of the features be assumed to be independent?
                                   Normally, this should be set to False as the features are never
                                   independent (e.g. mean and median)
    :type hypotheses_independent: bool
    :param n_jobs: Number of processes to use during the p-value calculation
    :type n_jobs: int
    :param show_warnings: Show warnings during the p-value calculation (needed for debugging of calculators).
    :type show_warnings: bool
    :param chunksize: The size of one chunk that is submitted to the worker
        process for the parallelisation.  Where one chunk is defined as
        the data for one feature. If you set the chunksize
        to 10, then it means that one task is to filter 10 features.
        If it is set it to None, depending on distributor,
        heuristics are used to find the optimal chunksize. If you get out of
        memory exceptions, you can try it with the dask distributor and a
        smaller chunksize.
    :type chunksize: None or int
    :return: A pandas.DataFrame with each column of the input DataFrame X as index with information on the significance
             of this particular feature. The DataFrame has the columns
             "feature",
             "type" (binary, real or const),
             "p_value" (the significance of this feature as a p-value, lower means more significant)
             "relevant" (True if the Benjamini Hochberg procedure rejected the null hypothesis [the feature is
             not relevant] for this feature).
             If the problem is `multiclass` with n classes, the DataFrame will contain n
             columns named "p_value_CLASSID" instead of the "p_value" column.
             `CLASSID` refers here to the different values set in `y`.
             There will also be n columns named `relevant_CLASSID`, indicating whether
             the feature is relevant for that class.
    :rtype: pandas.DataFrame
    """

    # Make sure X and y both have the exact same indices
    y = y.sort_index()
    X = X.sort_index()

    assert list(y.index) == list(X.index), "The index of X and y need to be the same"

    if ml_task not in ["auto", "classification", "regression"]:
        raise ValueError("ml_task must be one of: 'auto', 'classification', 'regression'")
    elif ml_task == "auto":
        ml_task = infer_ml_task(y)

    if multiclass:
        assert ml_task == "classification", "ml_task must be classification for multiclass problem"
        assert len(y.unique()) >= n_significant, "n_significant must not exceed the total number of classes"

        if len(y.unique()) <= 2:
            warnings.warn("Two or fewer classes, binary feature selection will be used (multiclass = False)")
            multiclass = False

    with warnings.catch_warnings():
        if not show_warnings:
            warnings.simplefilter("ignore")
        else:
            warnings.simplefilter("default")

        if n_jobs == 0 or n_jobs == 1:
            map_function = map
        else:
            pool = Pool(
                processes=n_jobs,
                initializer=initialize_warnings_in_workers,
                initargs=(show_warnings,),
            )
            map_function = partial(pool.map, chunksize=chunksize)

        relevance_table = pd.DataFrame(index=pd.Series(X.columns, name="feature"))
        relevance_table["feature"] = relevance_table.index
        relevance_table["type"] = pd.Series(
            map_function(get_feature_type, [X[feature] for feature in relevance_table.index]),
            index=relevance_table.index,
        )
        table_real = relevance_table[relevance_table.type == "real"].copy()
        table_binary = relevance_table[relevance_table.type == "binary"].copy()

        table_const = relevance_table[relevance_table.type == "constant"].copy()
        table_const["p_value"] = np.NaN
        table_const["relevant"] = False

        if not table_const.empty:

            warnings.warn(
                "[test_feature_significance] Constant features: {}".format(", ".join(map(str, table_const.feature))),
                RuntimeWarning,
            )

        if len(table_const) == len(relevance_table):
            if n_jobs < 0 or n_jobs > 1:
                pool.close()
                pool.terminate()
                pool.join()
            return table_const

        if ml_task == "classification":
            tables = []
            for label in y.unique():
                _test_real_feature = partial(
                    target_binary_feature_real_test,
                    y=(y == label),
                    test=test_for_binary_target_real_feature,
                )
                _test_binary_feature = partial(target_binary_feature_binary_test, y=(y == label))
                tmp = _calculate_relevance_table_for_implicit_target(
                    table_real,
                    table_binary,
                    X,
                    _test_real_feature,
                    _test_binary_feature,
                    hypotheses_independent,
                    fdr_level,
                    map_function,
                )
                if multiclass:
                    tmp = tmp.reset_index(drop=True)
                    tmp.columns = tmp.columns.map(
                        lambda x: x + "_" + str(label) if x != "feature" and x != "type" else x
                    )
                tables.append(tmp)

            if multiclass:
                relevance_table = reduce(
                    lambda left, right: pd.merge(left, right, on=["feature", "type"], how="outer"),
                    tables,
                )
                relevance_table["n_significant"] = relevance_table.filter(regex="^relevant_", axis=1).sum(axis=1)
                relevance_table["relevant"] = relevance_table["n_significant"] >= n_significant
                relevance_table.index = relevance_table["feature"]
            else:
                relevance_table = combine_relevance_tables(tables)

        elif ml_task == "regression":
            _test_real_feature = partial(target_real_feature_real_test, y=y)
            _test_binary_feature = partial(target_real_feature_binary_test, y=y)
            relevance_table = _calculate_relevance_table_for_implicit_target(
                table_real,
                table_binary,
                X,
                _test_real_feature,
                _test_binary_feature,
                hypotheses_independent,
                fdr_level,
                map_function,
            )

        if n_jobs < 0 or n_jobs > 1:
            pool.close()
            pool.terminate()
            pool.join()

        # set constant features to be irrelevant for all classes in multiclass case
        if multiclass:
            for column in relevance_table.filter(regex="^relevant_", axis=1).columns:
                table_const[column] = False
            table_const["n_significant"] = 0
            table_const.drop(columns=["p_value"], inplace=True)

        relevance_table = pd.concat([relevance_table, table_const], axis=0)

        if sum(relevance_table["relevant"]) == 0:
            warnings.warn(
                "No feature was found relevant for {} for fdr level = {} (which corresponds to the maximal percentage "
                "of irrelevant features, consider using an higher fdr level or add other features.".format(
                    ml_task, fdr_level
                ),
                RuntimeWarning,
            )

    return relevance_table


def _calculate_relevance_table_for_implicit_target(
    table_real,
    table_binary,
    X,
    test_real_feature,
    test_binary_feature,
    hypotheses_independent,
    fdr_level,
    map_function,
):
    table_real["p_value"] = pd.Series(
        map_function(test_real_feature, [X[feature] for feature in table_real.index]),
        index=table_real.index,
    )
    table_binary["p_value"] = pd.Series(
        map_function(test_binary_feature, [X[feature] for feature in table_binary.index]),
        index=table_binary.index,
    )
    relevance_table = pd.concat([table_real, table_binary])
    method = "fdr_bh" if hypotheses_independent else "fdr_by"
    relevance_table["relevant"] = multipletests(relevance_table.p_value, fdr_level, method)[0]
    return relevance_table.sort_values("p_value")


def infer_ml_task(y):
    """
    Infer the machine learning task to select for.
    The result will be either `'regression'` or `'classification'`.
    If the target vector only consists of integer typed values or objects, we assume the task is `'classification'`.
    Else `'regression'`.
    :param y: The target vector y.
    :type y: pandas.Series
    :return: 'classification' or 'regression'
    :rtype: str
    """
    if y.dtype.kind in np.typecodes["AllInteger"] or y.dtype == np.object:
        ml_task = "classification"
    else:
        ml_task = "regression"

    return ml_task


def combine_relevance_tables(relevance_tables):
    """
    Create a combined relevance table out of a list of relevance tables,
    aggregating the p-values and the relevances.
    :param relevance_tables: A list of relevance tables
    :type relevance_tables: List[pd.DataFrame]
    :return: The combined relevance table
    :rtype: pandas.DataFrame
    """

    def _combine(a, b):
        a.relevant |= b.relevant
        a.p_value = a.p_value.combine(b.p_value, min, 1)
        return a

    return reduce(_combine, relevance_tables)


def get_feature_type(feature_column):
    """
    For a given feature, determine if it is real, binary or constant.
    Here binary means that only two unique values occur in the feature.
    :param feature_column: The feature column
    :type feature_column: pandas.Series
    :return: 'constant', 'binary' or 'real'
    """
    n_unique_values = len(set(feature_column.values))
    if n_unique_values == 1:
        return "constant"
    elif n_unique_values == 2:
        return "binary"
    else:
        return "real"
