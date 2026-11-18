import numpy as np


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
