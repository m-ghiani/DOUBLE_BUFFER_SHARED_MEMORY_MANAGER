class LogMessages:
    """
    A class containing log messages for the DoubleBufferedSharedMemoryManager class.

    Attributes:
        WRONG_DATA_TYPE (str): The message to be displayed when the value is not an ndarray.
        WRONG_DATA_SIZE (str): The message to be displayed when the value size doesn't match the expected size.
        WRONG_DATA_SHAPE (str): The message to be displayed when the value shape doesn't match the expected shape.
    """
    WRONG_DATA_TYPE = "Value provided is of type {}, expected np.ndarray."
    WRONG_DATA_SIZE = "Value size mismatch: expected {} bytes, got {} bytes."
    WRONG_DATA_SHAPE = "Value shape mismatch: expected {}, got {}."
    WRONG_DATA_DTYPE = "Value dtype mismatch: expected {}, got {}."