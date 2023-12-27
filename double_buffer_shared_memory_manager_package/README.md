
# DoubleBufferedSharedMemoryManager

`DoubleBufferedSharedMemoryManager` is a Python class designed to manage double-buffered shared memory for concurrent read/write operations, particularly suited for handling video frames stored as NumPy ndarrays.

## Features

- **Double-Buffered Memory**: Allows one buffer to be read while the other is being written to, minimizing read-write conflicts.
- **Thread-Safe Access**: Utilizes reader-writer locks to allow multiple readers and a single writer.
- **Dynamic Memory Management**: Dynamically creates or connects to shared memory segments based on the provided base name.
- **Customizable Data Handling**: Supports specifying the shape and data type of the NumPy ndarray.

## Requirements

- Python 3.x
- NumPy
- multiprocessing
- readerwriterlock
- colorlog (for enhanced logging with color)

## Installation

Ensure you have the required libraries:

```bash
pip install numpy readerwriterlock colorlog
```

## Usage

### Initialization

Import and initialize the manager:

```python
from double_buffered_shared_memory import DoubleBufferedSharedMemoryManager
import numpy as np

# Initialize with default parameters
manager = DoubleBufferedSharedMemoryManager('image_buffer')

# Initialize with custom shape and dtype
manager = DoubleBufferedSharedMemoryManager('image_buffer', shape=(576, 720, 3), dtype=np.uint16)
```

### Writing to Shared Memory

```python
# Create a random HD image
img = np.random.randint(255, size=(1080, 1920, 3), dtype=np.uint8)

# Write to the active buffer
manager[0] = img
```

0 is a convention, you can use any number, but for clarity use 0

### Reading from Shared Memory

```python
# Switch the active buffer
manager.__switch()

# Read from the non-active buffer
img_copy = manager[0]
```

0 is a convention, you can use any number, but for clarity use 0

### Cleanup

It's important to clean up shared memory resources when they're no longer needed:

```python
manager.cleanup()
```

### Using with Context Manager

The class supports context management protocol for automatic resource management:

```python
with DoubleBufferedSharedMemoryManager('image_buffer') as manager:
    # Your code to use the manager
    pass
```

## Logging

The class utilizes `colorlog` for enhanced logging. Log messages will vary in color based on the log level to provide a clearer and more intuitive understanding of the operations and events.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/your-repo/issues) if you want to contribute.

## Author

Massimo Ghiani <m.ghiani@gmail.com>

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

- [NumPy](https://numpy.org/)
- [readerwriterlock](https://pypi.org/project/readerwriterlock/)
- [colorlog](https://pypi.org/project/colorlog/)
