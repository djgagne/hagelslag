#!/usr/bin/env python
import pygrib
import numpy as np
from os.path import exists
from pandas import DatetimeIndex
from .Grib_ModelGrid import Grib_ModelGrid

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

        if 'nam' in self.member:
            if '00' in self.member:
                for forecast_hr in self.forecast_hours:
                    file = '{1}/{2}/{3}/nam_conusnest_{3}00f0{0:02}.grib2'.format(forecast_hr,
                                                                                self.path,
                                                                                self.member,
                                                                                run_date.strftime("%Y%m%d"))
                    filenames.append(file)
            elif '12' in self.member:
                for forecast_hr in self.forecast_hours:
                    file = '{1}/{2}/{3}/nam_conusnest_{3}12f0{0:02}.grib2'.format(forecast_hr,
                                                                                self.path,
                                                                                self.member,
                                                                                run_date.strftime("%Y%m%d"))
                    filenames.append(file)
        else:
            member_name = str(self.member.split("_")[0])
            if '00' in self.member:
                for forecast_hr in self.forecast_hours:
                    file = '{1}/{2}/{3}/hiresw_conus{4}_{3}00f0{0:02}.grib2'.format(forecast_hr,
                                                                                    self.path,
                                                                                    self.member,
                                                                                    run_date.strftime("%Y%m%d"),
                                                                                    member_name)
                    filenames.append(file)

            elif '12' in self.member:
                for forecast_hr in self.forecast_hours:
                    file = '{1}/{2}/{3}/hiresw_conus{4}_{3}12f0{0:02}.grib2'.format(forecast_hr,
                                                                                    self.path,
                                                                                    self.member,
                                                                                    run_date.strftime("%Y%m%d"),
                                                                                    member_name)
                    filenames.append(file)

        super(HREFv2ModelGrid, self).__init__(filenames,run_date,start_date,end_date,variable,member)
        
        return 
