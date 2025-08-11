class SubmittedWeightsError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class LayerStateException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class MinerNotRegisteredException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class APIException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class NanInfWarning(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class NanInfException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class SpecVersionException(Exception):
    def __init__(self, expected_version: int, actual_version: str):
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(f"Spec version mismatch. Expected: {expected_version}, Received: {actual_version}")
