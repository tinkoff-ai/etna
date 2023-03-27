from typing import Set
from typing import Tuple

import pandas as pd


def find_columns_diff(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Tuple[Set[str], Set[str], Set[str]]:
    columns_before_transform = set(df_before.columns)
    columns_after_transform = set(df_after.columns)
    created_columns = columns_after_transform - columns_before_transform
    removed_columns = columns_before_transform - columns_after_transform

    columns_to_check_changes = columns_after_transform.intersection(columns_before_transform)
    changed_columns = set()
    for column in columns_to_check_changes:
        if not df_before[column].equals(df_after[column]):
            changed_columns.add(column)

    return created_columns, removed_columns, changed_columns
