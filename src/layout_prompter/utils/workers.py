import os
from typing import Final

# Define the maximum concurrency for langchain's chain.batch() execution.
# This number is set to 8 as increasing it further does not yield significant benefits especially in the data preprocessing (processor) stage.
MAX_CONCURRENCY: Final[int] = 8


def get_num_workers(max_concurrency: int = MAX_CONCURRENCY) -> int:
    """
    Calculate the number of workers based on the maximum concurrency.
    This function ensures that the number of workers does not exceed the maximum concurrency.

    Args:
        max_concurrency (int): The maximum number of concurrent workers allowed. Defaults to MAX_CONCURRENCY.

    Returns:
        int: The number of workers to be used, which is the minimum of max_concurrency and the number of available CPU cores.
    """

    # Get the number of available CPU cores
    # ref. https://stackoverflow.com/a/55423170
    max_cpu_cout = len(os.sched_getaffinity(0))

    # Ensure we do not exceed the number of available CPU cores
    return min(max_concurrency, max_cpu_cout)
