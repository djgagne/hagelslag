#!/usr/bin/env python
import numpy as np
from .GribModelGrid import GribModelGrid
from .ModelGrid import ModelGrid


class FV3ModelGrid(GribModelGrid):
    """
    Extension of the ModelGrid class for interfacing with the HREFv2  ensemble.
    Args:
        member (str): Name of the ensemble member
        run_date (datetime.datetime object): Date of the initial step of the ensemble run
        variable(int or str): name of grib2 variable(str) or grib2 message number(int) being loaded
        start_date (datetime.datetime object): First time step extracted.
        end_date (datetime.datetime object): Last time step extracted.
        path (str): Path to model output files
        single_step (boolean (default=True): Whether variable information is stored with each time step in a separate
                file (True) or one file containing all timesteps (False).
    """

    def __init__(self, member, run_date, variable, start_date, 
                end_date, path, single_step=True):
        self.path = path
        self.member = member
        filenames = []
        self.forecast_hours = np.arange((start_date - run_date).total_seconds() / 3600,
                                        (end_date - run_date).total_seconds() / 3600 + 1, dtype=int)
        
        for forecast_hr in self.forecast_hours:
            file_name=self.path+'{0}/{1}_{2}f{3:03d}.grib2'.format(
                                                        run_date.strftime("%Y%m%d"),
                                                        self.member,
                                                        run_date.strftime("%Y%m%d%H"),
                                                        forecast_hr)
            filenames.append(file_name)
        self.netcdf_variables = ["hske_1000", "hske_3000", "hmf_1000", "hmf_3000", "ihm_1000", "ihm_3000"]
        super(FV3ModelGrid, self).__init__(filenames, run_date, start_date, end_date, variable, member)
        
        return

    def load_data(self):
        if self.variable in self.netcdf_variables:
            return ModelGrid.load_data(self)
        else:
            return GribModelGrid.load_data(self)

