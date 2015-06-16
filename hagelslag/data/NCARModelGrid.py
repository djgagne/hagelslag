from ModelGrid import ModelGrid
import numpy as np
from pandas import DatetimeIndex

class NCARModelGrid(ModelGrid):
    def __init__(self, member, run_date, variable, start_date, end_date, path, single_step=True):
        self.member = member
        self.run_date = run_date
        self.variable = variable
        self.start_date = start_date
        self.end_date = end_date
        self.valid_dates = DatetimeIndex(start=start_date, end=end_date, freq="1H")
        self.path = path
        self.forecast_hours = np.arange((self.start_date - self.run_date).total_seconds() / 3600, 
                                        (self.end_date - self.run_date).total_seconds() / 3600 + 1)
        filenames = []
        if not single_step:
            filenames.append("{0}{1}/{2}_surrogate_{1}.nc".format(self.path, 
                                                                  self.run_date.strftime("%Y%m%d%H"), 
                                                                  self.member))
        else:
            for hour in self.forecast_hours:
                filenames.append("{0}{1}/post_rundir/{2}/fhr_{3:d}/WRFTWO{3:02d}.nc".format(self.path,
                                                                                            self.run_date.strftime("%Y%m%d%H"),
                                                                                            self.member,
                                                                                            hour))
        super(NCARModelGrid, self).__init__(filenames)

    def load_data(self):
        return super(NCARModelGrid, self).load_data(self.variable)

