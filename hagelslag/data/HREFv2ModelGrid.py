#!/usr/bin/env python
from .Grib_ModelGrid import Grib_ModelGrid
from datetime import timedelta 
import glob 
import numpy as np
import os

class HREFv2ModelGrid(Grib_ModelGrid):
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
                end_date, path,single_step=True):
        self.path = path
        self.member = member
        filenames = []
        self.forecast_hours = np.arange((start_date - run_date).total_seconds() / 3600,
                                        (end_date - run_date).total_seconds() / 3600 + 1, dtype=int)
        
        day_before_date = (run_date-timedelta(days=1)).strftime("%Y%m%d") 
        member_name = str(self.member.split("_")[0])
        if 'nam' in self.member:
            if '00' in self.member:
                for forecast_hr in self.forecast_hours:
                    files = glob.glob('{0}/{1}/nam_conusnest_*00*{2}*'.format(
                                    self.path,run_date.strftime("%y%m%d"),forecast_hr))
                    if not files: 
                        files = glob.glob('{0}/{1}/nam.t00z*{2}*'.format(
                                    self.path,run_date.strftime("%y%m%d"),forecast_hr))
                    filenames.append(files[0])
            elif '12' in self.member:
                for forecast_hr in self.forecast_hours:
                    files = glob.glob('{0}/{1}/nam_conusnest_*12*{2}*'.format(
                                    self.path,day_before_date,(forecast_hr+12)))
                    if not files: 
                        files = glob.glob('{0}/{1}/nam.t12z*{2}*'.format(
                                    self.path,day_before_date,(forecast_hr+12)))
                    filenames.append(files[0])
        else:
            if '00' in self.member:
                for forecast_hr in self.forecast_hours:
                    files = glob.glob('{0}/{1}/hiresw_conus*{2}*00*{3}*'.format(
                            self.path,run_date.strftime("%y%m%d"),member_name,forecast_hr))
                    filenames.append(files[0])
            elif '12' in self.member:
                for forecast_hr in self.forecast_hours:
                    files = glob.glob('{0}/{1}/hiresw_conus*{2}*12*{3}*'.format(
                            self.path,day_before_date,member_name,(forecast_hr+12)))
                    filenames.append(files[0])
        super(HREFv2ModelGrid, self).__init__(filenames,run_date,start_date,end_date,variable,member)
        return 
