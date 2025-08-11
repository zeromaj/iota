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
