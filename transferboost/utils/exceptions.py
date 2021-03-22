class NotFittedError(Exception):
    """Error to be returned when the model is not fitted."""

    def __init__(self, message):
        """Error constructor.

        Args:
            message (string): error message
        """
        self.message = message


class DimensionalityError(Exception):
    """Error to be returned when the array dimensionality is wrong."""

    def __init__(self, message):
        """Error constructor.

        Args:
            message (string): error message
        """
        self.message = message


class UnsupportedModelError(Exception):
    """Error to be Model is not supported by the package."""

    def __init__(self, message):
        """Error constructor.

        Args:
            message (string): error message
        """
        self.message = message
