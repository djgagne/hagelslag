from .ModelGrid import ModelGrid
import numpy as np
import os
from os.path import exists


class SSEFModelGrid(ModelGrid):
    """
    Extension of ModelGrid to the CAPS Storm-Scale Ensemble Forecast system.

    Args:
        member (str): Name of the ensemble member
        run_date (datetime.datetime object): Date of the initial step of the ensemble run
        start_date (datetime.datetime object): First time step extracted.
        end_date (datetime.datetime object): Last time step extracted.
        path (str): Path to model output files.
        single_step (boolean (default=False)): Whether variable information is stored with each time step in a separate
            file or one file containing all timesteps.
    """
    def __init__(self, member, run_date, variable, start_date, end_date, path, single_step=False):
        self.path = path
        self.member = member
        forecast_hours = np.arange((start_date - run_date).total_seconds() / 3600,
                                   (end_date - run_date).total_seconds() / 3600 + 1)
        if single_step:
            full_path = self.path + "/".join([member, run_date.strftime("%Y%m%d"), "0000Z", "data2d"]) + "/"
            if not exists(full_path):
                full_path = self.path + "/".join([run_date.strftime("%Y%m%d"), member, "0000Z", "data2d"]) + "/"
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
