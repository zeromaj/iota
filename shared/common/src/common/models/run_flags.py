from pydantic import BaseModel


class RunFlag(BaseModel):
    enabled: bool = False
    version: float = 1.0

    def isOn(self) -> bool:
        return bool(self.enabled)

    def isOff(self) -> bool:
        return not self.isOn()

    def isExactlyOn(self, expected_version: int) -> bool:
        return bool(self.enabled and self.version == expected_version)

    def isAtLeastOn(self, expected_version: int) -> bool:
        return bool(self.enabled and self.version >= expected_version)


class RunFlags(BaseModel):
    test_flag: RunFlag = RunFlag()
    compress_s3_files: RunFlag = RunFlag()
    async_activation_cache: RunFlag = RunFlag()


RUN_FLAGS = RunFlags()
