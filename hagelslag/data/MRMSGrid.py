from netCDF4 import Dataset, num2date
import pandas as pd
import numpy as np
import os

class MRMSGrid(object):
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
        data = []
        valid_dates = []
        mrms_files = np.array(sorted(os.listdir(self.path + self.variable)))
        mrms_file_dates = pd.DatetimeIndex([m_file.split("_")[-2].split("-")[0]
            for m_file in mrms_files])
        for timestamp in self.all_dates:
            mrms_file = mrms_files[timestamp.date() == mrms_file_dates.date]
            file_obj = Dataset(self.path + self.variable + "/" + mrms_file)
            file_valid_dates = pd.DatetimeIndex(num2date(file_obj.variables["date"][:],
                                                         file_obj.variables["date"].units))
            time_index = np.where(file_valid_dates == timestamp)[0]
            if len(time_index) > 0:
                data.append(file_obj.variables[self.variable][time_index])
                valid_dates.append(timestamp)
        self.data = np.array(data)
        self.valid_dates = pd.DatetimeIndex(valid_dates)
            
    
