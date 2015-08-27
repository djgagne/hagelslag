import numpy as np
import pandas as pd
from hagelslag.data.MRMSGrid import MRMSGrid
from netCDF4 import Dataset, num2date
from datetime import timedelta


class NeighborEvaluator(object):
    def __init__(self, run_date, start_hour, end_hour, model_name, forecast_variable, mrms_variable,
                 neighbor_radii, smoothing_radii, size_thresholds, obs_mask, mask_variable,
                 forecast_path, mrms_path):
        self.run_date = run_date
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.model_name = model_name
        self.forecast_variable = forecast_variable
        self.mrms_variable = mrms_variable
        self.obs_mask = obs_mask
        self.mask_variable = mask_variable
        self.neighbor_radii = neighbor_radii
        self.smoothing_radii = smoothing_radii
        self.size_thresholds = size_thresholds
        self.forecast_path = forecast_path
        self.mrms_path = mrms_path
        self.hourly_forecasts = {}
        self.period_forecasts = {}
        self.raw_obs = {}
        self.period_obs = {}

    def load_forecasts(self):
        run_date_str = self.run_date.strftime("%Y%m%d")
        forecast_file = self.forecast_path + "/{0}/{1}_{2}_consensus_{0}.nc".format(run_date_str, self.model_name,
                                                                                    self.forecast_variable)
        forecast_data = Dataset(forecast_file)
        for size_threshold in self.size_thresholds:
            for smoothing_radii in self.smoothing_radii:
                for neighbor_radii in self.neighbor_radii:
                    hour_var = "neighbor_prob_r_{0:d}_s_{1:d}_{2}_{3:0.2f}".format(neighbor_radii, smoothing_radii,
                                                                              self.forecast_variable,
                                                                              size_threshold)
                    period_var = "neighbor_prob_{0:d}-hour_r_{1:d}_s_{2:d}_{4}_{4:0.2f}".format(self.end_hour -
                                                                                                self.start_hour + 1,
                                                                                                neighbor_radii,
                                                                                                smoothing_radii,
                                                                                                self.forecast_variable,
                                                                                                size_threshold)
                    self.hourly_forecasts[hour_var] = forecast_data.variables[hour_var][:]
                    self.period_forecasts[period_var] = forecast_data.variables[period_var][:]
        forecast_data.close()

    def load_obs(self,  mask_threshold=0.5):
        """
        Loads observations and masking grid (if needed).

        :param mask_threshold: Values greater than the threshold are kept, others are masked.
        :return:
        """
        start_date = self.run_date + timedelta(hours=self.start_hour)
        end_date = self.run_date + timedelta(hours=self.end_hour)
        mrms_grid = MRMSGrid(start_date, end_date, self.mrms_variable, self.mrms_path)
        mrms_grid.load_data()
        if len(mrms_grid.data) > 0:
            self.raw_obs[self.mrms_variable] = np.where(mrms_grid.data > 100, 100, mrms_grid.data)
            self.period_obs[self.mrms_variable] = self.raw_obs[self.mrms_variable].max(axis=0)
            if self.obs_mask:
                mask_grid = MRMSGrid(start_date, end_date, self.mask_variable, self.mrms_path)
                mask_grid.load_data()
                self.raw_obs[self.mask_variable] = np.where(mask_grid.data >= mask_threshold, 1, 0)
                self.period_obs[self.mask_variable] = self.raw_obs[self.mask_variable].max(axis=0)