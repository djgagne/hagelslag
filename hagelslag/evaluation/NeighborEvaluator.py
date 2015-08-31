import numpy as np
import pandas as pd
from hagelslag.data.MRMSGrid import MRMSGrid
from hagelslag.evaluation.ProbabilityMetrics import DistributedReliability, DistributedROC
from netCDF4 import Dataset, num2date
from datetime import timedelta
from scipy.signal import fftconvolve
from skimage.morphology import disk


class NeighborEvaluator(object):
    """
    A framework for statistically evaluating neighborhood probability forecasts.

    Parameters
    ----------
    run_date : datetime.datetime object
        Date of the beginning of the model run
    start_hour : int
        First forecast hour evaluated
    end_hour : int
        Last forecast hour evaluated
    model_name : str
        Name of the ensemble or machine learning model being evaluated
    forecast_variable : str
        Name of the forecast variable being evaluated.
    mrms_variable : str
        Name of the NSSL MRMS product being used for gridded observations
    neighbor_radii : list or array of ints
        neighborhood radii in number of grid points
    smoothing_radii : list or array of ints
        radius of Gaussian filter used by the forecast
    size_thresholds : list or array of floats
        Intensity threshold for neighborhood probabilities
    obs_mask : boolean
        Whether or not another MRMS product is used to mask invalid grid points
    mask_variable : str
        MRMS variable used for masking invalid grid points
    forecast_path : str
        Path to forecast files
    mrms_path : str
        Path to MRMS data
    """
    def __init__(self, run_date, start_hour, end_hour, model_name, forecast_variable, mrms_variable,
                 neighbor_radii, smoothing_radii, size_thresholds, probability_levels, obs_mask, mask_variable,
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
        self.probability_levels = probability_levels
        self.forecast_path = forecast_path
        self.mrms_path = mrms_path
        self.hourly_forecasts = {}
        self.period_forecasts = {}
        self.raw_obs = {}
        self.period_obs = {}

    def load_forecasts(self):
        run_date_str = self.run_date.strftime("%Y%m%d")
        forecast_file = self.forecast_path + "{0}/{1}_{2}_consensus_{0}.nc".format(run_date_str, self.model_name,
          self.forecast_variable)
        print "forecast file", forecast_file
        forecast_data = Dataset(forecast_file)
        for size_threshold in self.size_thresholds:
            for smoothing_radius in self.smoothing_radii:
                for neighbor_radius in self.neighbor_radii:
                    hour_var = "neighbor_prob_r_{0:d}_s_{1:d}_{2}_{3:0.2f}".format(neighbor_radius, smoothing_radius,
                                                                                   self.forecast_variable,
                                                                                   float(size_threshold))
                    period_var = "neighbor_prob_{0:d}-hour_r_{1:d}_s_{2:d}_{3}_{4:0.2f}".format(self.end_hour -
                                                                                                self.start_hour + 1,
                                                                                                neighbor_radius,
                                                                                                smoothing_radius,
                                                                                                self.forecast_variable,
                                                                                                float(size_threshold))
                    
                    print "Loading forecasts ", self.run_date, self.model_name, self.forecast_variable, size_threshold,
                    print smoothing_radius
                    self.hourly_forecasts[hour_var] = forecast_data.variables[hour_var][:]
                    self.period_forecasts[period_var] = forecast_data.variables[period_var][:]
        forecast_data.close()

    def load_obs(self,  mask_threshold=0.5):
        """
        Loads observations and masking grid (if needed).

        :param mask_threshold: Values greater than the threshold are kept, others are masked.
        :return:
        """
        print "Loading obs ", self.run_date, self.model_name, self.forecast_variable
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

    def evaluate_hourly_forecasts(self):
        score_columns = ["Run_Date", "Forecast_Hour", "Model_Name", "Forecast_Variable", "Neighbor_Radius",
                         "Smoothing_Radius", "Size_Threshold",  "ROC", "Reliability"]
        all_scores = pd.DataFrame(columns=score_columns)
        for h, hour in enumerate(range(self.start_hour, self.end_hour + 1)):
            for neighbor_radius in self.neighbor_radii:
                n_filter = disk(neighbor_radius)
                for size_threshold in self.size_thresholds:
                    print "Eval hour forecast ", hour, self.model_name, self.forecast_variable, self.run_date,
                    print neighbor_radius, size_threshold
                    hour_obs = fftconvolve(self.raw_obs[self.mrms_variable][h] >= size_threshold, n_filter, mode="same")
                    hour_obs[hour_obs > 1] = 1
                    if self.obs_mask:
                        hour_obs = hour_obs[self.raw_obs[self.mask_variable][h] > 0]
                    for smoothing_radius in self.smoothing_radii:
                        hour_var = "neighbor_prob_r_{0:d}_s_{1:d}_{2}_{3:0.2f}".format(neighbor_radius,
                                                                                       smoothing_radius,
                                                                                       self.forecast_variable,
                                                                                       size_threshold)
                        if self.obs_mask:
                            hour_forecast = self.hourly_forecasts[hour_var][h][self.raw_obs[self.mask_variable][h] > 0]
                        else:
                            hour_forecast = self.hourly_forecasts[hour_var][h]
                        roc = DistributedROC(thresholds=self.probability_levels, obs_threshold=0.5)
                        roc.update(hour_forecast, hour_obs)
                        rel = DistributedReliability(thresholds=self.probability_levels, obs_threshold=0.5)
                        rel.update(hour_forecast, hour_obs)
                        row = [self.run_date, hour, self.model_name, self.forecast_variable, neighbor_radius,
                               smoothing_radius, size_threshold, roc, rel]
                        all_scores.loc[hour_var + "_{0:d}".format(hour)] = row
        return all_scores

    def evaluate_period_forecasts(self):
        score_columns = ["Run_Date", "Model_Name", "Forecast_Variable", "Neighbor_Radius",
                         "Smoothing_Radius", "Size_Threshold",  "ROC", "Reliability"]
        all_scores = pd.DataFrame(columns=score_columns)

        for neighbor_radius in self.neighbor_radii:
            n_filter = disk(neighbor_radius)
            for size_threshold in self.size_thresholds:
                period_obs = fftconvolve(self.period_obs[self.mrms_variable] >= size_threshold, n_filter, mode="same")
                period_obs[period_obs > 1] = 1
                if self.obs_mask:
                    period_obs = period_obs[self.period_obs[self.mask_variable] > 0]
                else:
                    period_obs = period_obs.ravel()
                for smoothing_radius in self.smoothing_radii:
                    print "Eval period forecast ", self.model_name, self.forecast_variable, self.run_date,
                    print neighbor_radius, size_threshold, smoothing_radius
                    period_var = "neighbor_prob_{0:d}-hour_r_{1:d}_s_{2:d}_{3}_{4:0.2f}".format(self.end_hour -
                                                                                                self.start_hour + 1,
                                                                                                neighbor_radius,
                                                                                                smoothing_radius,
                                                                                                self.forecast_variable,
                                                                                                size_threshold)
                    if self.obs_mask:
                        period_forecast = self.period_forecasts[period_var][self.period_obs[self.mask_variable] > 0]
                    else:
                        period_forecast = self.period_forecasts[period_var].ravel()
                    roc = DistributedROC(thresholds=self.probability_levels, obs_threshold=0.5)
                    roc.update(period_forecast, period_obs)
                    rel = DistributedReliability(thresholds=self.probability_levels, obs_threshold=0.5)
                    rel.update(period_forecast, period_obs)
                    row = [self.run_date, self.model_name, self.forecast_variable, neighbor_radius,
                           smoothing_radius, size_threshold, roc, rel]
                    all_scores.loc[period_var] = row
        return all_scores


