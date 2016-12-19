from netCDF4 import Dataset
import pygrib
import numpy as np
import pandas as pd


class HRRRModelGrid(object):
    def __init__(self, run_date, variables, start_date, end_date, path, grib=True, frequency="1H"):
        self.run_date = pd.Timestamp(run_date)
        self.variables = variables
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.path = path
        self.valid_dates = pd.DatetimeIndex(start=self.start_date,
                                            end=self.end_date,
                                            freq=self.frequency)
        self.grib = grib
        self.data = {}
        self.units = {}
        self.long_names = {}

    def load_data(self):
        if self.grib:
            pass
        else:
            pass

    def load_hrrr_grib_table(self):
        pass

    def to_netcdf(self, out_path):
        pass

