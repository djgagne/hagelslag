#!/usr/bin/env python
from .GribModelGrid import GribModelGrid
from datetime import timedelta 
import numpy as np
from glob import glob


class HREFv2ModelGrid(GribModelGrid):
    """
    Extension of the ModelGrid class for interfacing with the HREFv2  ensemble.
    Args:
        member (str): Name of the ensemble member
        run_date (datetime.datetime object): Date of the initial step of the ensemble run
        variable(int or str): name of grib2 variable(str) or grib2 message number(int) being loaded
        start_date (datetime.datetime object): First time step extracted.
        end_date (datetime.datetime object): Last time step extracted.
        path (str): Path to model output files
    """

    def __init__(self, member, run_date, variable, start_date, 
                end_date, path):
        self.path = path
        self.member = member
        filenames = []
        self.forecast_hours = np.arange((start_date - run_date).total_seconds() / 3600,
                                        (end_date - run_date).total_seconds() / 3600 + 1, dtype=int)
        
        day_before_date = (run_date-timedelta(days=1)).strftime("%Y%m%d") 
        member_name = str(self.member.split("_")[0])
        if '00' in self.member:
            inilization='00'
            hours = self.forecast_hours
            date = run_date.strftime("%Y%m%d")
        elif '12' in self.member:
            inilization='12'
            hours = self.forecast_hours+12
            date = day_before_date
        for forecast_hr in hours:
            if 'nam' in self.member:
                files = glob('{0}/{1}/nam*conusnest*{2}f*{3}*'.format(self.path,
                        date,inilization,forecast_hr))
                if not files:
                    files = glob('{0}/{1}/nam*t{2}z*conusnest*{3}*'.format(self.path,
                            date,inilization,forecast_hr))
            else:
                files = glob('{0}/{1}/*hiresw*conus{2}*{3}f*{4}*'.format(self.path,
                        date,member_name,inilization,forecast_hr))
            if len(files) >=1:
                filenames.append(files[0])
        super(HREFv2ModelGrid, self).__init__(filenames,run_date,start_date,end_date,variable,member)
        return 
