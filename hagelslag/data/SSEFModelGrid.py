from ModelGrid import ModelGrid
from glob import glob
import pandas as pd
import numpy as np
import os

class SSEFModelGrid(ModelGrid):
    """

    """
    def __init__(self, member, run_date, variable, start_date, end_date, path, single_step=False):
        self.path = path
        self.member = member
        forecast_hours = np.arange((start_date - run_date).total_seconds() / 3600,
                                   (end_date - run_date).total_seconds() / 3600 + 1)
        if single_step:
            full_path = self.path + "/".join([member, run_date.strftime("%Y%m%d"), "0000Z", "data2d"]) + "/"
        else:
            full_path = self.path + "/".join([member, run_date.strftime("%Y%m%d")]) + "/"
        potential_filenames = []
        if single_step:
            for hour in forecast_hours:
                potential_filenames.append("{0}ar{1}00.net{2}{3:06d}".format(full_path, 
                                                                             run_date.strftime("%Y%m%d"),
                                                                             variable.ljust(6,"_"),
                                                                             int(hour) * 3600))
        else:
            potential_filenames.append("{0}ssef_{1}_{2}_{3}.nc".format(full_path,
                                                                       self.member,
                                                                       run_date.strftime("%Y%m%d"),
                                                                       variable))
        filenames = []
        for filename in potential_filenames:
            if os.access(filename, os.R_OK):
                filenames.append(filename)
        super(SSEFModelGrid, self).__init__(filenames, run_date, start_date, end_date, variable)
        return
