import math
from common import settings as common_settings


def calculate_n_partitions(n_miners: int) -> int:
    """Calculate the number of partitions for a given number of miners.
    #TODO: 5x but maybe needs more thinking.

    Args:
        n_miners (int): The number of miners.

    Returns:
        int: The number of partitions.
    """
    # return int(max(1, n_miners * (n_miners - 1) / 2))
    return n_miners * 5


def calculate_num_parts(data: bytes) -> int:
    """Calculate the number of parts to upload a file to S3.

    Args:
        data (bytes): The data to upload.

    Returns:
        int: The number of parts to upload.
    """
    return int(math.ceil(len(data) / common_settings.MAX_PART_SIZE))
