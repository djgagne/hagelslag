import numpy as np
import pandas as pd
from hagelslag.data.ModelGrid import ModelGrid
from os.path import join


class HRRRModelGrid(ModelGrid):
    def __init__(self, run_date, variable, start_date, end_date, path, frequency="1H"):
        self.run_date = pd.Timestamp(run_date)
        self.variable = variable
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.frequency = frequency
        self.path = path
        self.valid_dates = pd.DatetimeIndex(start=self.start_date,
                                            end=self.end_date,
                                            freq=self.frequency)
        self.forecast_hours = np.array((self.valid_dates - self.run_date).total_seconds() / 3600, dtype=int)
        filenames = []
        for forecast_hour in self.forecast_hours:
            filenames.append(join(self.path, self.run_date.strftime("%Y%m%d"),
                                  "{0}_f{1:03d}_HRRR.nc4".format(self.run_date.strftime("%Y%m%d_i%H"),
                                                                 forecast_hour)))

        super(HRRRModelGrid, self).__init__(filenames, run_date, start_date, end_date, variable)


