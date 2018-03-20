from netCDF4 import Dataset
import numpy as np
from hagelslag.data.MRMSGrid import MRMSGrid
from hagelslag.evaluation.ProbabilityMetrics import DistributedROC, DistributedReliability
from datetime import timedelta
from scipy.ndimage import binary_dilation


class GridEvaluator(object):
    """
    An evaluation system for gridded forecasts.

    GridEvaluator loads in a set of machine learning model forecasts from a single model run, loads in corresponding
    observations, and then generates verification statistics from the matching of forecasts and observations. Forecasts
    can be aggregated in time with flexible window sizes.

    Parameters
    ----------
    run_date : datetime.datetime object
        The date of an ensemble run.
    ensemble_name : str
        Name of the ensemble. Should be consistent with the name used in the data processing and forecasting.
    ensemble_member : str
        Name of the ensemble member being loaded.
    model_names : list of strings
        Names of the machine learning models being evaluated.
    size_thresholds : list or numpy.ndarray of ints
        Intensity thresholds at which probability forecasts are made.
    start_hour : int
        Forecast hour at which evaluation begins.
    end_hour : int
        Forecast hour at which evaluation ends, inclusive.
    window_size : int
        Number of hours to include within a forecast window.
    time_skip : int
        Number of hours to skip between window starts
    forecast_path : str
        Path to where gridded forecasts are located.
    mrms_path : str
        Path to the MRMS gridded observations.
    mrms_variable: str
        Name of the variable being used for verification.
    obs_mask : bool, optional (default=True)
        Whether or not a masking grid is used to determine which grid points are evaluated
    mask_variable : str, optional (default="RadarQualityIndex_00.00")
        Name of the MRMS variable used for masking.
    """
    def __init__(self, run_date, ensemble_name, ensemble_member,
                 model_names, size_thresholds, start_hour, end_hour, window_size, time_skip,
                 forecast_path, mrms_path, mrms_variable, obs_mask=True,
                 mask_variable="RadarQualityIndex_00.00"):
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
        self.forecast_path = forecast_path
        self.mrms_path = mrms_path
        self.mrms_variable = mrms_variable
        self.raw_forecasts = {}
        self.raw_obs = {}
        self.window_forecasts = {}
        self.window_obs = {}
        self.dilated_obs = {}
        self.obs_mask = obs_mask
        self.mask_variable = mask_variable

    def load_forecasts(self):
        """
        Load the forecast files into memory.
        """
        run_date_str = self.run_date.strftime("%Y%m%d")
        for model_name in self.model_names:
            self.raw_forecasts[model_name] = {}
            forecast_file = self.forecast_path + run_date_str + "/" + \
                model_name.replace(" ", "-") + "_hailprobs_{0}_{1}.nc".format(self.ensemble_member, run_date_str)
            forecast_obj = Dataset(forecast_file)
            forecast_hours = forecast_obj.variables["forecast_hour"][:]
            valid_hour_indices = np.where((self.start_hour <= forecast_hours) & (forecast_hours <= self.end_hour))[0]
            for size_threshold in self.size_thresholds:
                self.raw_forecasts[model_name][size_threshold] = \
                    forecast_obj.variables["prob_hail_{0:02d}_mm".format(size_threshold)][valid_hour_indices]
            forecast_obj.close()

    def get_window_forecasts(self):
        """
        Aggregate the forecasts within the specified time windows.
        """
        for model_name in self.model_names:
            self.window_forecasts[model_name] = {}
            for size_threshold in self.size_thresholds:
                self.window_forecasts[model_name][size_threshold] = \
                    np.array([self.raw_forecasts[model_name][size_threshold][sl].sum(axis=0)
                              for sl in self.hour_windows])

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
            self.window_obs[self.mrms_variable] = np.array([self.raw_obs[self.mrms_variable][sl].max(axis=0)
                                                            for sl in self.hour_windows])
            if self.obs_mask:
                mask_grid = MRMSGrid(start_date, end_date, self.mask_variable, self.mrms_path)
                mask_grid.load_data()
                self.raw_obs[self.mask_variable] = np.where(mask_grid.data >= mask_threshold, 1, 0)
                self.window_obs[self.mask_variable] = np.array([self.raw_obs[self.mask_variable][sl].max(axis=0)
                                                               for sl in self.hour_windows])

    def dilate_obs(self, dilation_radius):
        """
        Use a dilation filter to grow positive observation areas by a specified number of grid points

        :param dilation_radius: Number of times to dilate the grid.
        :return:
        """
        for s in self.size_thresholds:
            self.dilated_obs[s] = np.zeros(self.window_obs[self.mrms_variable].shape)
            for t in range(self.dilated_obs[s].shape[0]):
                self.dilated_obs[s][t][binary_dilation(self.window_obs[self.mrms_variable][t] >= s, iterations=dilation_radius)] = 1

    def roc_curves(self, prob_thresholds):
        """
        Generate ROC Curve objects for each machine learning model, size threshold, and time window.

        :param prob_thresholds: Probability thresholds for the ROC Curve
        :param dilation_radius: Number of times to dilate the observation grid.
        :return: a dictionary of DistributedROC objects.
        """
        all_roc_curves = {}
        for model_name in self.model_names:
            all_roc_curves[model_name] = {}
            for size_threshold in self.size_thresholds:
                all_roc_curves[model_name][size_threshold] = {}
                for h, hour_window in enumerate(self.hour_windows):
                    hour_range = (hour_window.start, hour_window.stop)
                    all_roc_curves[model_name][size_threshold][hour_range] = \
                        DistributedROC(prob_thresholds, 1)
                    if self.obs_mask:
                        all_roc_curves[model_name][size_threshold][hour_range].update(
                            self.window_forecasts[model_name][size_threshold][h][
                                self.window_obs[self.mask_variable][h] > 0],
                            self.dilated_obs[size_threshold][h][self.window_obs[self.mask_variable][h] > 0]
                        )
                    else:
                        all_roc_curves[model_name][size_threshold][hour_range].update(
                            self.window_forecasts[model_name][size_threshold][h],
                            self.dilated_obs[size_threshold][h]
                        )
        return all_roc_curves

    def reliability_curves(self, prob_thresholds):
        """
        Output reliability curves for each machine learning model, size threshold, and time window.

        :param prob_thresholds:
        :param dilation_radius:
        :return:
        """
        all_rel_curves = {}
        for model_name in self.model_names:
            all_rel_curves[model_name] = {}
            for size_threshold in self.size_thresholds:
                all_rel_curves[model_name][size_threshold] = {}
                for h, hour_window in enumerate(self.hour_windows):
                    hour_range = (hour_window.start, hour_window.stop)
                    all_rel_curves[model_name][size_threshold][hour_range] = \
                        DistributedReliability(prob_thresholds, 1)
                    if self.obs_mask:
                        all_rel_curves[model_name][size_threshold][hour_range].update(
                            self.window_forecasts[model_name][size_threshold][h][
                                self.window_obs[self.mask_variable][h] > 0],
                            self.dilated_obs[size_threshold][h][self.window_obs[self.mask_variable][h] > 0]
                        )
                    else:
                        all_rel_curves[model_name][size_threshold][hour_range].update(
                            self.window_forecasts[model_name][size_threshold][h],
                            self.dilated_obs[size_threshold][h]
                        )
        return all_rel_curves








