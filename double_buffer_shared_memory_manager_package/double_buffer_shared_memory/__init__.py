"""
This package provides a DoubleBufferedSharedMemoryManager class for managing
double-buffered shared memory, suitable for concurrent read/write operations video frames data stored as a NumPy ndarray.
"""

from .double_buffered_shared_memory_manager import DoubleBufferedSharedMemoryManager

__version__ = '1.0.0'
__author__ = 'Massimo Ghiani <m.ghiani@gmail.com>'
__status__ = 'Production'

__all__ = ['DoubleBufferedSharedMemoryManager']
