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
        mrms_files = np.array(sorted(os.listdir(self.path + self.variable + "/")))
        mrms_file_dates = np.array([m_file.split("_")[-2].split("-")[0]
            for m_file in mrms_files])
        mrms_file = None
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
        self.data = np.array(data)
        self.valid_dates = pd.DatetimeIndex(valid_dates) 
