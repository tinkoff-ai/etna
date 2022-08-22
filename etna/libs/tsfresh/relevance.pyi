from typing import List
from typing import Optional

import pandas as pd

def calculate_relevance_table(
    X: pd.DataFrame, 
    y: pd.Series, 
    ml_task: str = ..., 
    multiclass: bool = ..., 
    n_significant: int = ..., 
    n_jobs: int = ..., 
    show_warnings: bool = ..., 
    chunksize: Optional[int] = ..., 
    test_for_binary_target_binary_feature: str = ...,
    test_for_binary_target_real_feature: str = ...,
    test_for_real_target_binary_feature: str = ..., 
    test_for_real_target_real_feature: str = ..., 
    fdr_level: float = ..., 
    hypotheses_independent: bool = ...,
    ) -> pd.DataFrame: ...

def infer_ml_task(y: pd.Series) -> str: ...
def combine_relevance_tables(relevance_tables: List[pd.DataFrame]) -> pd.DataFrame: ...
def get_feature_type(feature_column: pd.Series) -> str: ...
