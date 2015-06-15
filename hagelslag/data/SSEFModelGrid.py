from ModelGrid import ModelGrid
from glob import glob
import pandas as pd
import numpy as np


class SSEFModelGrid(ModelGrid):
    def __init__(self, member, run_date, variable, start_date, end_date, path, single_step=False):
        self.path = path
        self.member = member
        self.run_date = run_date
        self.variable = variable
        self.start_date = start_date
        self.end_date = end_date
        self.valid_dates = DatetimeIndex(start=self.start_date, end=self.end_date, freq="1H")
        self.forecast_hours = np.arange((self.start_date - self.run_date).total_seconds() / 3600, 
                                        (self.end_date - self.run_date).total_seconds() / 3600 + 1)
        if single_step:
            full_path = self.path + "/".join([member, run_date.strftime("%Y%m%d"), "0000Z", "data2d"]) + "/"
        else:
            full_path = self.path + "/".join([member, run_date.strftime("%Y%m%d")]) + "/"
        filenames = []
        if single_step:
            for hour in self.forecast_hours:
                filenames.append("{0}ar{1}00.net{2}{3:06d}".format(full_path, 
                                                                   self.run_date.strftime("%Y%m%d"),
                                                                   self.variable.ljust(6,"_"),
                                                                   hour * 3600))
        else:
            filenames.append("{0}ssef_{1}_{2}_{3}.nc".format(full_path,
                                                             self.member,
                                                             self.run_date.strftime("%Y%m%d"),
                                                             self.variable))      
        super(SSEFModelGrid, self).__init__(filenames) 
        return

    def load_data(self):
        return super(SSEFModelGrid, self).load_data(self.variable)
