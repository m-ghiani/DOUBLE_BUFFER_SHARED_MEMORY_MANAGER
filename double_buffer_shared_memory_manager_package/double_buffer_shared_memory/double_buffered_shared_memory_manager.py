from multiprocessing import shared_memory
from readerwriterlock import rwlock
import numpy as np
import logging
from colorlog import ColoredFormatter
from double_buffer_shared_memory.log_messages import LogMessages


class DoubleBufferedSharedMemoryManager:
    """
    Manages double-buffered shared memory using reader-writer locks to allow multiple readers
    and a single writer. It's specifically designed for managing video frames data stored as a NumPy ndarray.

    Attributes:
        name_base (str): The base name for the shared memory segments.
        shape (tuple): The shape of the ndarray, default is (1080, 1920, 3) for an HD image.
        dtype (data-type): The desired data-type for the array, default is np.uint8.
        size (int): The total size in bytes calculated from shape and dtype.
        lock (RWLockFair): A fair reader-writer lock for managing access to the shared memory.
        buffers (list): A list containing two shared memory segments.
        active_index (int): The index of the currently active buffer for writing.

    Methods:
        __repr__: Returns a representation of the object for debugging.
        __str__: Returns a string representation of the object.
        __int__: Returns the size of the shared memory.
        __len__: Returns the total size in bytes of the ndarray.
        __switch: Switches the active buffer index to the other buffer (internal use).
        __getitem__: Reads the entire buffer from the non-active shared memory and returns it as an ndarray.
        __setitem__: Writes the entire ndarray to the active shared memory.
        __enter__: Context management protocol - returns itself as a context manager.
        __exit__: Context management protocol - cleans up the shared memory resources.
        cleanup: Cleans up the shared memory resources manually.

    Example:
        >>> manager = DoubleBufferedSharedMemoryManager('image_buffer')
        >>> img = np.random.randint(255, size=(1080, 1920, 3), dtype=np.uint8)
        >>> manager[0] = img  # Write to the active buffer
        >>> manager.__switch()  # Switch the active buffer
        >>> img_copy = manager[0]  # Read from the non-active buffer
    """

    __doc__ = "Class to manage double-buffered shared memory for concurrent read/write operations."
    __defaults__ = ("name_base", "size", "6220800")  # Default values for the parameters
    __author__ = "Massimo Ghiani <m.ghiani@gmail.com>"
    __status__ = "production"
    # The following module attributes are no longer updated.
    __version__ = "1.0.0"
    __date__ = "27 December 2023"
    __maintainer__ = "Massimo Ghiani <m.ghiani@gmail.com>"

    def __init__(
        self,
        name_base,
        shape=(1080, 1920, 3),
        dtype=np.uint8,
        create_on_enter=False,
        cleanup=True,
    ):
        """
        Initializes a DoubleBufferedSharedMemoryManager object with specified name base, shape, and data type.

        Parameters:
            name_base (str): The base name for the shared memory segments.
            shape (tuple): The shape of the ndarray, default is (1080, 1920, 3) for an HD image.
            dtype (data-type): The desired data-type for the array, default is np.uint8.
            create_on_enter (bool): Whether to create the shared memory segments when entering the context.
            cleanup (bool): Whether to clean up the shared memory resources when exiting the context.

        Example:
            >>> manager = DoubleBufferedSharedMemoryManager('image_buffer')
            >>> manager = DoubleBufferedSharedMemoryManager('image_buffer', shape=(576,720,3))
            >>> manager = DoubleBufferedSharedMemoryManager('image_buffer', shape=(576,720,3), dtype=np.uint16)
            >>> manager = DoubleBufferedSharedMemoryManager('image_buffer', shape=(576,720,3), dtype=np.uint16, create_on_enter=True)
            >>> manager = DoubleBufferedSharedMemoryManager('image_buffer', shape=(576,720,3), dtype=np.uint16, create_on_enter=True, cleanup=False)
        """
        self.name_base = name_base
        self.shape = shape
        self.dtype = dtype
        self.size = np.prod(shape) * np.dtype(dtype).itemsize
        self.__cleanup_on_exit = cleanup
        self.__lock = rwlock.RWLockFair()
        self.__create_on_enter = create_on_enter
        self.buffers = []
        self.__init_logger()

        # Attempt to create or connect to the shared memory segments
        if not create_on_enter:
            self.__init_shared_memory()

        self.active_index = 0
        self.__logger.info("DoubleBufferedSharedMemoryManager initialized")

    def __init_logger(self):
        """
        Initializes the logger.
        This method is intended for internal use only.
        """
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__logger.setLevel(logging.INFO)
        log_colors = {
            "DEBUG": "green",
            "INFO": "cyan",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        }
        formatter = ColoredFormatter(
            "%(log_color)s%(levelname)s%(reset)s - %(asctime)s - %(name)s - %(funcName)s - line %(lineno)d - %(message)s",
            log_colors=log_colors,
            secondary_log_colors={},
            style="%",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.__logger.addHandler(handler)

    def __init_shared_memory(self):
        """
        Initializes the shared memory segments.
        This method is intended for internal use only.
        """
        for i in range(2):
            name = f"{self.name_base}_{i}"
            shm_created = False
            try:
                shm = shared_memory.SharedMemory(name=name)
                if shm.size < self.size:
                    self.__logger.warning("Shared memory size mismatch, unlinking...")
                    shm.close()
                    shm.unlink()
                else:
                    self.__logger.info(f"Connected to shared memory block {name}")
                    self.buffers.append(shm)
                    shm_created = True
            except FileNotFoundError:
                self.__logger.info(f"No existing shared memory block found with name {name}.")
            except Exception as e:
                self.__logger.error(f"Error accessing shared memory block {name}: {e}")

            if not shm_created:
                self.__logger.info(f"Creating shared memory block {name}")
                shm = shared_memory.SharedMemory(name=name, create=True, size=self.size)
                self.buffers.append(shm)

    def __repr__(self):
        """
        Returns a string representation of the object for debugging.

        Returns:
            str: A string representation of the DoubleBufferedSharedMemoryManager.

        Example:
            >>> manager = DoubleBufferedSharedMemoryManager('image_buffer')
            >>> repr(manager)
            "<DoubleBufferedSharedMemoryManager(name_base='image_buffer', size=6220800, active_index=0)>"
        """
        return (
            f"<DoubleBufferedSharedMemoryManager(name_base={self.name_base}, "
            f"size={self.size}, active_index={self.active_index})>"
        )

    def __str__(self):
        """
        Returns a string representation of the object.

        Returns:
            str: A user-friendly string representation of the DoubleBufferedSharedMemoryManager.

        Example:
            >>> manager = DoubleBufferedSharedMemoryManager('image_buffer')
            >>> print(manager)
            "DoubleBufferedSharedMemoryManager with name base image_buffer and size 6220800 bytes. Currently active buffer index: 0"
        """
        return (
            f"DoubleBufferedSharedMemoryManager with "
            f"name base {self.name_base} and "
            f"size {self.size} bytes. Currently active buffer index: {self.active_index}"
        )

    def __int__(self):
        """
        Returns the size of the shared memory as an integer.

        Returns:
            int: The size of the shared memory.

        Example:
            >>> manager = DoubleBufferedSharedMemoryManager('image_buffer')
            >>> int(manager)
            6220800
        """
        return self.size

    def __len__(self):
        """
        Returns the total size in bytes of the ndarray.

        Returns:
            int: The total size in bytes.

        Example:
            >>> manager = DoubleBufferedSharedMemoryManager('image_buffer')
            >>> len(manager)
            6220800
        """
        return self.size

    def __switch(self):
        """
        Switches the active buffer index to the other buffer.
        This method is intended for internal use only.
        """
        with self.__lock.gen_wlock():
            self.active_index = 1 - self.active_index

    def __getitem__(self, _):
        """
        Reads the entire buffer from the non-active shared memory and returns it as an ndarray.

        Parameters:
            _ (int): A placeholder for the key parameter, not used in this method.

        Returns:
            np.ndarray: The image data read from the non-active shared memory.

        Example:
            >>> manager = DoubleBufferedSharedMemoryManager('image_buffer')
            >>> img = np.random.randint(255, size=(1080, 1920, 3), dtype=np.uint8)
            >>> manager[0] = img  # Write to the active buffer
            >>> manager.__switch()  # Switch the active buffer
            >>> img_copy = manager[0]  # Read from the non-active buffer
        """
        with self.__lock.gen_rlock():
            read_index = 1 - self.active_index
            data = self.buffers[read_index].buf[: self.size]
        return np.frombuffer(data, dtype=np.uint8).reshape((1080, 1920, 3))

    def __setitem__(self, _, value: np.ndarray):
        """
        Writes the entire ndarray to the active shared memory.

        Parameters:
            _ (int): A placeholder for the key parameter, not used in this method.
            value (np.ndarray): The image data to be written into the active shared memory.

        Raises:
            ValueError: If the value is not an ndarray with the specified size, shape, and dtype.
        """
        # Verifica che il valore sia un ndarray
        if not isinstance(value, np.ndarray):
            self.__logger.error(LogMessages.WRONG_DATA_TYPE.format(type(value)))
            raise TypeError(LogMessages.WRONG_DATA_TYPE.format(type(value)))

        # Calcola la dimensione attesa e confrontala con la dimensione dell'ndarray fornito
        expected_size = self.size
        actual_size = value.size * value.itemsize
        if actual_size != expected_size:
            self.__logger.error(
                LogMessages.WRONG_DATA_SIZE.format(expected_size, actual_size)
            )
            raise ValueError(
                LogMessages.WRONG_DATA_SIZE.format(expected_size, actual_size)
            )

        # Verifica che la forma dell'ndarray corrisponda a quella prevista
        if value.shape != self.shape:
            self.__logger.error(
                LogMessages.WRONG_DATA_SHAPE.format(self.shape, value.shape)
            )
            raise ValueError(
                LogMessages.WRONG_DATA_SHAPE.format(self.shape, value.shape)
            )

        # Verifica che il dtype dell'ndarray corrisponda a quello previsto
        if value.dtype != self.dtype:
            self.__logger.error(
                LogMessages.WRONG_DATA_DTYPE.format(self.dtype, value.dtype)
            )
            raise ValueError(
                LogMessages.WRONG_DATA_DTYPE.format(self.dtype, value.dtype)
            )

        with self.__lock.gen_wlock():
            data = value.tobytes()
            self.buffers[self.active_index].buf[: self.size] = data
            self.__switch()

    def __enter__(self):
        """
        Context management protocol - returns itself as a context manager.

        Returns:
            DoubleBufferedSharedMemoryManager: The instance itself.

        Example:
            >>> manager = DoubleBufferedSharedMemoryManager('image_buffer')
            >>> img = np.random.randint(255, size=(1080, 1920, 3), dtype=np.uint8)
            >>> manager[0] = img  # Write to the active buffer
        """
        if self.__create_on_enter:
            self.__init_shared_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context management protocol - cleans up the shared memory resources when exiting the context.

        Parameters:
            exc_type: The exception type if an exception was raised in the context.
            exc_val: The exception value if an exception was raised.
            exc_tb: The traceback if an exception was raised.

        Example:
            >>> with DoubleBufferedSharedMemoryManager('image_buffer') as manager:
            ...     pass  # Your code to use the manager
        """
        if self.__cleanup_on_exit:
            self.cleanup()

    def cleanup(self):
        """
        Cleans up the shared memory resources manually.
        This method should be called to ensure that the shared memory is properly cleaned up when it's no longer needed.

        Example:
            >>> manager = DoubleBufferedSharedMemoryManager('image_buffer')
            >>> manager.cleanup()  # Clean up resources when done
        """
        for buffer in self.buffers:
            buffer.close()
            if buffer.name:
                buffer.unlink()
