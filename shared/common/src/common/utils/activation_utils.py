import hashlib


def generate_uid_from_index(idx: int, secret: str) -> str:
    """
    Generates a unique ID from an index and a secret.

    Args:
        idx (int): The index of the sample to generate.
        secret (str): The master access key to use for the activation.

    Returns:
        str: The uid of the sample.
    """

    return hashlib.sha256((str(idx) + secret).encode()).hexdigest()
