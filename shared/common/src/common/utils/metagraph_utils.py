def is_validator(network: str, validator_permit: bool, stake: float, vpermit_rao_limit: int = 10_000) -> bool:
    """Checks if a UID on the subnet is a validator."""
    if network == "test":
        return stake >= vpermit_rao_limit
    else:
        return validator_permit and stake >= vpermit_rao_limit
