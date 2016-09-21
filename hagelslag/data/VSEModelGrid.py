from ModelGrid import ModelGrid
from glob import glob
import pandas as pd
import numpy as np
import os, pdb
from os.path import exists

class VSEModelGrid(ModelGrid):
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
        full_path = self.path + "/".join([member, run_date.strftime("%Y%m%d%H"), "post_AGAIN"]) + "/"
        potential_filenames = []
        for hour in forecast_hours:
            potential_filenames.append("{0}fhr_{1:d}/WRFTWO{2:02d}.nc".format(full_path, 
                                                                         int(hour), int(hour) ))
        filenames = []
        for filename in potential_filenames:
            if os.access(filename, os.R_OK):
                filenames.append(filename)
        super(VSEModelGrid, self).__init__(filenames, run_date, start_date, end_date, variable)
        return