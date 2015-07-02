from netCDF4 import Dataset
import numpy as np
from hagelslag.data.MRMSGrid import MRMSGrid
from datetime import timedelta

class GridEvaluator(object):
    def __init__(self, run_date, ensemble_name, ensemble_member,
                 model_names, size_thresholds, start_hour, end_hour, window_size, time_skip,
                 forecast_sample_path, mrms_path, mrms_variable):
        self.run_date = run_date
        self.ensemble_name = ensemble_name
        self.ensemble_member = ensemble_member
        self.model_names = model_names
        self.size_thresholds = size_thresholds
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.valid_hours = np.arange(self.start_hour, self.end_hour + 1)
        self.window_size = window_size
        self.time_skip = time_skip
        self.hour_windows = []
        for hour in self.valid_hours[self.window_size-1::self.time_skip]:
            self.hour_windows.append(slice(hour-self.window_size+1 - self.start_hour,
                                           hour+1-self.start_hour))
        self.forecast_sample_path = forecast_sample_path
        self.mrms_path = mrms_path
        self.mrms_variable = mrms_variable
        self.raw_forecasts = {}
        self.raw_obs = None
        self.window_forecasts = {}
        self.window_obs = None

    def load_forecasts(self):
        run_date_str = self.run_date.strftime("%Y%m%d")
        for model_name in self.model_names:
            self.raw_forecasts[model_name] = {}
            forecast_file = self.forecast_sample_path + run_date_str + "/" + \
                model_name.replace(" ", "-") + "_hailprobs_{0}_{1}.nc".format(self.ensemble_member, run_date_str)
            forecast_obj = Dataset(forecast_file)
            forecast_hours = forecast_obj.variables["forecast_hour"][:]
            valid_hour_indices = np.where((self.start_hour <= forecast_hours) & (forecast_hours <= self.end_hour))[0]
            for size_threshold in self.size_thresholds:
                self.raw_forecasts[model_name][size_threshold] = \
                    forecast_obj.variables["prob_hail_{0:02d}_mm".format(size_threshold)][valid_hour_indices]
            forecast_obj.close()

    def get_window_forecasts(self):
        for model_name in self.model_names:
            self.window_forecasts[model_name] = {}
            for size_threshold in self.size_thresholds:
                self.window_forecasts[model_name][size_threshold] = \
                    np.array([self.raw_forecasts[model_name][size_threshold][sl].sum(axis=0)
                              for sl in self.hour_windows])

    def load_obs(self, rqi_mask=True, rqi_threshold=0.5):
        start_date = self.run_date + timedelta(hours=self.start_hour)
        end_date = self.run_date + timedelta(hours=self.end_hour)
        mrms_grid = MRMSGrid(start_date, end_date, self.mrms_variable, self.mrms_path)
        mrms_grid.load_data()
        if len(mrms_grid.data) > 0:
            if rqi_mask:
                rqi_grid = MRMSGrid(start_date, end_date, "RadarQualityIndex_00.00", self.mrms_path)
                rqi_grid.load_data()
            self.raw_obs = mrms_grid.data

