import numpy as np
import pandas as pd
from pandas import date_range
from os.path import exists, join
import s3fs
import xarray as xr
from datetime import timedelta


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
        file_objects (list): List of the file objects for each model time step
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
        self.valid_dates = date_range(start=self.start_date,
                                         end=self.end_date,
                                         freq=self.frequency)
        print(self.run_date)
        print(type(self.run_date))
        self.forecast_hours = (self.valid_dates - self.run_date).astype("timedelta64[h]").astype(int)


    def load_data(self):

        units = ""
        level = self.variable.split('-')[1]
        self.variable = self.variable.split('-')[0]
        fs = s3fs.S3FileSystem(anon=True)
        files = []
        run_date_str = self.run_date.strftime("%Y%m%d")
        forecast_hour = self.run_date.strftime("%H")
        path = join(self.path, run_date_str, f'{run_date_str}_{forecast_hour}z_fcst.zarr', level, self.variable, level)
        f = s3fs.S3Map(root=path, s3=fs, check=False)
        files.append(f)

        ds = xr.open_mfdataset(files, engine='zarr', parallel=True).load()
        arr = ds[self.variable].values.astype('float32')
        dummy_forecast_hour_00 = np.zeros((1, arr.shape[1], arr.shape[2]))
        array = np.concatenate([dummy_forecast_hour_00, arr])[self.forecast_hours[0]:self.forecast_hours[-1] + 1, :, :]

        if hasattr(ds[self.variable], 'units'):
            units = ds[self.variable].attrs['units']

        return array, units
