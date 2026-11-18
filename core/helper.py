import numpy as np


def is_power_of_n_or_zero(x: int, n: int) -> bool:
    """
    Returns True if x is 0 or an exact power of n (n^0, n^1, n^2, ...),
    with n >= 2.
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    if x == 0:
        return True

    if x < 0:
        return False

    while x % n == 0:
        x //= n

    return x == 1


def find_zero_chunks(array):
    """
    Find consecutive rows of all zeros in an N*3 array.

    Args:
        array (numpy.ndarray): Input N*3 array.

    Returns:
        list: List of [start_index, end_index] for each chunk of consecutive zero rows.
    """
    zero_rows = np.all(array == 0, axis=1)  # Identify rows that are all zero
    chunks = []
    start = None

    for i, is_zero in enumerate(zero_rows):
        if is_zero and start is None:
            start = i
        elif not is_zero and start is not None:
            chunks.append([start, i - 1])
            start = None

    if start is not None:
        chunks.append([start, len(zero_rows) - 1])

    return chunks
