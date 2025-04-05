"""Logs stuff from the IVP thing"""

import numpy as np
from numba import float64, int64    # import the types
from numba.experimental import jitclass

spec = [
    ('data', float64[:,:]),
    ('data_columns', int64),
    ('valid_data_size', int64)
]

@jitclass(spec)
class IVPLogger(object):
    """Jitted Logger Object to Run in the Function as an arg"""

    def __init__(self, data_columns):
        self.data_columns = data_columns
        self.data = np.zeros((1,data_columns))
        self.valid_data_size = 0

    def increase_size(self):
        """Double the data size if necessary"""
        new_size_of_data = (np.size(self.data)//self.data_columns) * 2
        self.data = np.resize(self.data, (new_size_of_data, self.data_columns))

    def append_data(self, new_line):
        """Append a new line of data"""          
        if self.valid_data_size + 1 > np.size(self.data) // self.data_columns:
            self.increase_size()

        self.data[self.valid_data_size, :] = new_line

        self.valid_data_size = self.valid_data_size + 1

    def trim_excess(self):
        """Trim excess data"""
        self.data = np.resize(self.data, (self.valid_data_size, self.data_columns))

    def return_data(self):
        """Returns the whole data array"""
        self.trim_excess()
        return self.data
    
    def return_data_size(self):
        """Returns the size of data"""
        self.trim_excess()
        return self.valid_data_size
