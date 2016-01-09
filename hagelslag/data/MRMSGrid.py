from netCDF4 import Dataset, num2date
import pandas as pd
import numpy as np
import os


class MRMSGrid(object):
    """
    An interface to the NOAA National Severe Storms Lab Multi-Radar Multi-Sensor (MRMS) dataset.

    MRMSGrid assumes that the data are in netCDF format and have been interpolated to match the grid being used for
    forecasting.

    Args:
        start_date (datetime.datetime or time str): Date of first time step to be loaded.
        end_date (datetime.datetime or str in timestamp format): Date of last time step to be loaded.
        variable (str): MRMS variable name
        path (str): Path to the directory containing MRMS files.
        freq (str, optional (default="1H")): Time frequency of the data being loaded. Uses pandas time syntax.

    Attributes:
        start_date (datetime.datetime or time str): Date of first time step to be loaded.
        end_date (datetime.datetime or str in timestamp format): Date of last time step to be loaded.
        variable (str): MRMS variable name
        path (str): Path to the directory containing MRMS files.
        freq (str, optional (default="1H")): Time frequency of the data being loaded. Uses pandas time syntax.
        all_dates : pandas.DatetimeIndex
        List of dates being loaded
        data (ndarray or None): Array of gridded observations after load_data is called. None otherwise.
        valid_dates (ndarray): Contains the dates where data loaded successfully.
    """
    def __init__(self, start_date, end_date, variable, path, freq="1H"):
        self.start_date = start_date
        self.end_date = end_date
        self.variable = variable
        self.path = path
        self.freq = freq
        self.all_dates = pd.DatetimeIndex(start=start_date, end=end_date, freq=freq)
        self.data = None
        self.valid_dates = None

    def load_data(self):
        """
        Loads data files and stores the output in the data attribute.
        """
        data = []
        valid_dates = []
        mrms_files = np.array(sorted(os.listdir(self.path + self.variable + "/")))
        mrms_file_dates = np.array([m_file.split("_")[-2].split("-")[0]
            for m_file in mrms_files])
        old_mrms_file = None
        file_obj = None
        for t in range(self.all_dates.shape[0]):
            file_index = np.where(mrms_file_dates == self.all_dates[t].strftime("%Y%m%d"))[0]
            if len(file_index) > 0:
                mrms_file = mrms_files[file_index][0]
                if mrms_file != old_mrms_file:
                    if file_obj is not None:
                        file_obj.close()
                    file_obj = Dataset(self.path + self.variable + "/" + mrms_file)
                    old_mrms_file = mrms_file
                    
                    if "time" in file_obj.variables.keys():
                        time_var = "time"
                    else:
                        time_var = "date"
                    file_valid_dates = pd.DatetimeIndex(num2date(file_obj.variables[time_var][:],
                                                                 file_obj.variables[time_var].units))
                time_index = np.where(file_valid_dates.values == self.all_dates.values[t])[0]
                if len(time_index) > 0:
                    data.append(file_obj.variables[self.variable][time_index[0]])
                    valid_dates.append(self.all_dates[t])
        if file_obj is not None:
            file_obj.close()
        self.data = np.array(data)
        self.data[self.data < 0] = 0
        self.data[self.data > 100] = 100
        self.valid_dates = pd.DatetimeIndex(valid_dates) 
