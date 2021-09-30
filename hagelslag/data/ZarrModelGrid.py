from os.path import join

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from pandas import date_range


class ZarrModelGrid(object):
    """
    Base class for reading 2D model output grids from HRRR Zarr data streamed off of AWS.

    Given an AWS bucket name, loads the values of a single variable from a model run. Supports model output in
    Zarr format.

    Attributes:
        path (str): Base Path for AWS Bucket
        run_date (ISO date string or datetime.datetime object): Date of the initialization time of the model run.
        start_date (ISO date string or datetime.datetime object): Date of the first timestep extracted.
        end_date (ISO date string or datetime.datetime object): Date of the last timestep extracted.
        freqency (str): spacing between model time steps.
        valid_dates: DatetimeIndex of all model timesteps
        forecast_hours: array of all hours in the forecast
    """

    def __init__(self,
                 path,
                 run_date,
                 start_date,
                 end_date,
                 variable,
                 frequency="1H"):
        self.path = path
        self.variable = variable
        self.run_date = pd.to_datetime(run_date)
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.frequency = frequency
        self.valid_dates = date_range(start=self.start_date, end=self.end_date, freq=self.frequency)
        self.forecast_hours = (self.valid_dates - self.run_date).astype("timedelta64[h]").astype(int)

    def load_data(self):

        units = ""
        level = self.variable.split('-')[1]
        self.variable = self.variable.split('-')[0]
        fs = s3fs.S3FileSystem(anon=True)
        run_date_str = self.run_date.strftime("%Y%m%d")
        run_hour = self.run_date.strftime("%H")
        path = join(self.path, run_date_str, f'{run_date_str}_{run_hour}z_fcst.zarr', level, self.variable, level)
        f = s3fs.S3Map(root=path, s3=fs, check=False)
        ds = xr.open_mfdataset([f], engine='zarr', parallel=True).load()

        if self.run_date in self.valid_dates:
            arr = ds[self.variable].values[self.forecast_hours[0]:self.forecast_hours[-1] + 1].astype('float32')
            forecast_hour_00_path = join(self.path, run_date_str, f'{run_date_str}_{run_hour}z_anl.zarr', level,
                                         self.variable.replace('1hr_', ''), level)
            fh_0_file = s3fs.S3Map(root=forecast_hour_00_path, s3=fs, check=False)
            fh_0_ds = xr.open_mfdataset([fh_0_file], engine='zarr', parallel=True).expand_dims('time')
            fh_0_arr = fh_0_ds[self.variable.replace('1hr_', '')].values
            array = np.concatenate([fh_0_arr, arr])[self.forecast_hours[0]:self.forecast_hours[-1] + 1, :, :]
        else:
            array = ds[self.variable].values[self.forecast_hours[0] - 1:self.forecast_hours[-1]].astype('float32')

        if hasattr(ds[self.variable], 'units'):
            units = ds[self.variable].attrs['units']

        return array, units
