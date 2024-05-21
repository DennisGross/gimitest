import os
import numpy as np
import pandas as pd

from typing import Optional, List, Dict, Union


class Logger:
    '''
    A class for logging data values to a .txt file in a specific format.

    Args:
    - filepath (str): The path to the log file.

    Usage:
    - Initialize the logger with a file path.
    - Write the header with the write_columns method before logging.
    - Use the log method to add a new log entry.
    - Use load_logs to load logs into a Pandas DataFrame.
    '''

    def __init__(self, filepath: str, columns: List[str] = None, delimiter: str = '; ') -> None:
        self.delimiter = delimiter

        if columns is None:
            assert os.path.isfile(filepath)
            with open(filepath, 'r') as file:
                header_line = file.readline().strip()
                columns = header_line.split(self.delimiter)
            # print('No columns provided; found {} columns in the file'.format(len(columns)))

        assert len(columns) > 0
        assert np.all([isinstance(c, str) for c in columns])

        self.filepath = filepath
        self.columns = columns.copy() # type: List[str]
        self.n = len(self.columns)


    def write_columns(self):
        with open(self.filepath, 'w') as file:
            file.write(self.delimiter.join(self.columns) + '\n')


    def log(self, **kwargs) -> None:
        '''Serializes and appends the data to log.'''
        data_serialized = {k: self._serialize(kwargs.get(k, None)) for k in self.columns}
        self._log(data_serialized)


    def load_logs(self) -> pd.DataFrame:
        '''
        Load logs from the file and return as a Pandas DataFrame.

        Returns:
        - pd.DataFrame: DataFrame containing the logged data.
        '''
        data = []
        with open(self.filepath, 'r') as file:
            header_line = file.readline().strip()
            assert header_line.split(self.delimiter) == self.columns, header_line.split(self.delimiter)
            for line in file:
                values = [v.strip() for v in line.strip().split(self.delimiter)]
                assert len(values) == self.n
                data.append([self._deserialize(v) if v != 'None' else None for v in values])

        return pd.DataFrame(data, columns=self.columns)


    def _log(self, data: Dict[str, str]) -> None:
        '''Appends a line to the log file.'''
        log_line = self.delimiter.join([data[k] for k in self.columns])

        with open(self.filepath, 'a') as file:
            file.write(log_line + '\n')


    def _serialize(self, data = None) -> str:
        if data is None:
            return 'None'
        elif isinstance(data, np.ndarray):
            return np.array2string(data, separator=',').replace('\n', '')
        else:
            return str(data)


    def _deserialize(self, data: str) -> Union[np.ndarray, float, bool]:
        if data.startswith('['):
            return np.array(eval('np.array(' + data + ')'))
        elif data.endswith('e'):
            return data == 'True'
        else:
            return float(data)
