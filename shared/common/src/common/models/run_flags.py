from pydantic import BaseModel


class RunFlag(BaseModel):
    enabled: bool = False
    version: float = 1.0


class RunFlags(BaseModel):
    test_flag: RunFlag = RunFlag()


def isOn(flag: RunFlag) -> bool:
    """Return True if the flag exists and is enabled."""
    return bool(flag and flag.enabled)


def isOff(flag: RunFlag) -> bool:
    """Return True if the flag is missing or explicitly disabled."""
    return not isOn(flag)


def isExactlyOn(flag: RunFlag, expected_version: int) -> bool:
    """True if flag is enabled and version == expected_version"""
    return bool(flag and flag.enabled and flag.version == expected_version)


def isAtLeastOn(flag: RunFlag, min_version: int) -> bool:
    """True if flag is enabled and version >= min_version"""
    return bool(flag and flag.enabled and flag.version >= min_version)
