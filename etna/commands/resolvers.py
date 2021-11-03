from typing import List


def shift(steps_number: int, values: List[int]) -> List[int]:
    """Shift values for steps_number steps."""
    return [v + steps_number for v in values]
