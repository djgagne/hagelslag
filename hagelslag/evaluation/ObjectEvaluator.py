import numpy as np
import pandas as pd
import json
from glob import glob
from hagelslag.evaluation.ProbabilityMetrics import DistributedCRPS, DistributedReliability, DistributedROC
from scipy.stats import gamma


class ObjectEvaluator(object):
    """
    ObjectEvaluator performs a statistical evaluation of object-based severe weather forecasts.

    ObjectEvaluator loads forecast and observation files for a particular ensemble member and model run and then matches
    the forecasts with their assigned observations. Verification statistics can be calculated on the full dataset
    or on subsets selected based on filter queries.

    Attributes:
        run_date (datetime.datetime): The date marking the start of the model run.
        ensemble_name (str): The name of the ensemble or NWP model being used.
        ensemble_member (str): The name of the ensemble member being evaluated.
        model_names (list): The names of the machine learning models being evaluated
        model_types (list): The types of machine learning models being evaluated.
        forecast_bins (dict of str and numpy.ndarray pairs): For machine learning models forecasting a discrete pdf,
            this specifies the bin labels used.
        dist_thresholds (array): Thresholds used to discretize probability distribution forecasts.
        forecast_json_path (str): Full path to the directory containing all json files with the forecast values.
        track_data_csv_path (str): Full path to the directory containing the csv data files used for training.
        metadata_columns (list): Columns pulled from track data csv files.
        type_cols (dict): Map between forecast type used in json files and observation column in csv files
        forecasts (dict): Dictionary of DataFrames containing forecast information from csv files
        matched_forecasts (dict): Forecasts merged with observation information.

    """
    def __init__(self, run_date, ensemble_name, ensemble_member, model_names, model_types, forecast_bins,
                 dist_thresholds, forecast_json_path, track_data_csv_path):
        self.run_date = run_date
        self.ensemble_name = ensemble_name
        self.ensemble_member = ensemble_member
        self.model_names = model_names
        self.model_types = model_types
        self.forecast_bins = forecast_bins
        self.dist_thresholds = dist_thresholds
        self.forecast_json_path = forecast_json_path
        self.track_data_csv_path = track_data_csv_path
        self.metadata_columns = ["Track_ID", "Obs_Track_ID", "Ensemble_Name", "Ensemble_Member", "Forecast_Hour",
                                 "Step_Duration", "Total_Duration", "Area"]
        self.type_cols = {"size": "Hail_Size",
                          "translation-x": "Translation_Error_X",
                          "translation-y": "Translation_Error_Y",
                          "start-time": "Start_Time_Error",
                          "dist": ["Shape", "Location", "Scale"],
                          "condition": "Hail_Size"}
        self.forecasts = {}
        self.obs = None
        self.matched_forecasts = {}
        for model_type in self.model_types:
            self.forecasts[model_type] = {}
            for model_name in self.model_names[model_type]:
                self.forecasts[model_type][model_name] = pd.DataFrame(columns=self.metadata_columns +
                                                                      list(self.forecast_bins[model_type].astype(str)))

    def load_forecasts(self):
        """
        Loads the forecast files and gathers the forecast information into pandas DataFrames.
        """
        forecast_path = self.forecast_json_path + "/{0}/{1}/".format(self.run_date.strftime("%Y%m%d"),
                                                                     self.ensemble_member)
        forecast_files = sorted(glob(forecast_path + "*.json"))
        for forecast_file in forecast_files:
            file_obj = open(forecast_file)
            json_obj = json.load(file_obj)
            file_obj.close()
            track_id = json_obj['properties']["id"]
            obs_track_id = json_obj['properties']["obs_track_id"]
            forecast_hours = json_obj['properties']['times']
            duration = json_obj['properties']['duration']
            for f, feature in enumerate(json_obj['features']):
                area = np.sum(feature["properties"]["masks"])
                step_id = track_id + "_{0:02d}".format(f)
                for model_type in self.model_types:
                    for model_name in self.model_names[model_type]:
                        prediction = feature['properties'][model_type + "_" + model_name.replace(" ", "-")]
                        if model_type == "condition":
                            prediction = [prediction]
                        row = [track_id, obs_track_id, self.ensemble_name, self.ensemble_member, forecast_hours[f],
                               f + 1, duration, area] + prediction
                        self.forecasts[model_type][model_name].loc[step_id] = row

    def load_obs(self):
        """
        Loads the track total and step files and merges the information into a single data frame.
        """
        track_total_file = self.track_data_csv_path + \
            "track_total_{0}_{1}_{2}.csv".format(self.ensemble_name,
                                                 self.ensemble_member,
                                                 self.run_date.strftime("%Y%m%d"))
        track_step_file = self.track_data_csv_path + \
            "track_step_{0}_{1}_{2}.csv".format(self.ensemble_name,
                                                self.ensemble_member,
                                                self.run_date.strftime("%Y%m%d"))
        track_total_cols = ["Track_ID", "Translation_Error_X", "Translation_Error_Y", "Start_Time_Error"]
        track_step_cols = ["Step_ID", "Track_ID", "Hail_Size", "Shape", "Location", "Scale"]
        track_total_data = pd.read_csv(track_total_file, usecols=track_total_cols)
        track_step_data = pd.read_csv(track_step_file, usecols=track_step_cols)
        obs_data = pd.merge(track_step_data, track_total_data, on="Track_ID", how="left")
        self.obs = obs_data

    def merge_obs(self):
        """
        Match forecasts and observations.
        """
        for model_type in self.model_types:
            self.matched_forecasts[model_type] = {}
            for model_name in self.model_names[model_type]:
                self.matched_forecasts[model_type][model_name] = pd.merge(self.forecasts[model_type][model_name],
                                                                          self.obs, right_on="Step_ID", how="left",
                                                                          left_index=True)

    def crps(self, model_type, model_name, condition_model_name, condition_threshold, query=None):
        """
        Calculates the cumulative ranked probability score (CRPS) on the forecast data.

        Args:
            model_type: model type being evaluated.
            model_name: machine learning model being evaluated.
            condition_model_name: Name of the hail/no-hail model being evaluated
            condition_threshold: Threshold for using hail size CDF
            query: pandas query string to filter the forecasts based on the metadata


        Returns:
            a DistributedCRPS object
        """

        def gamma_cdf(x, a, loc, b):
            if a == 0 or b == 0:
                cdf = np.ones(x.shape)
            else:
                cdf = gamma.cdf(x, a, loc, b)
            return cdf

        crps_obj = DistributedCRPS(self.dist_thresholds)
        if query is not None:
            sub_forecasts = self.matched_forecasts[model_type][model_name].query(query)
            sub_forecasts = sub_forecasts.reset_index(drop=True)
            condition_forecasts = self.matched_forecasts["condition"][condition_model_name].query(query)
            condition_forecasts = condition_forecasts.reset_index(drop=True)
        else:
            sub_forecasts = self.matched_forecasts[model_type][model_name]
            condition_forecasts = self.matched_forecasts["condition"][condition_model_name]
        if sub_forecasts.shape[0] > 0:
            if model_type == "dist":
                forecast_cdfs = np.zeros((sub_forecasts.shape[0], self.dist_thresholds.size))
                for f in range(sub_forecasts.shape[0]):
                    condition_prob = condition_forecasts.loc[f, self.forecast_bins["condition"][0]]
                    if condition_prob >= condition_threshold:
                        f_params = [0, 0, 0]
                    else:
                        f_params = sub_forecasts[self.forecast_bins[model_type]].values[f]
                    forecast_cdfs[f] = gamma_cdf(self.dist_thresholds, f_params[0], f_params[1], f_params[2])
                obs_cdfs = np.array([gamma_cdf(self.dist_thresholds, *params)
                                    for params in sub_forecasts[self.type_cols[model_type]].values])
                crps_obj.update(forecast_cdfs, obs_cdfs)
            else:
                crps_obj.update(sub_forecasts[self.forecast_bins[model_type].astype(str)].values,
                                sub_forecasts[self.type_cols[model_type]].values)

        return crps_obj

    def roc(self, model_type, model_name, intensity_threshold, prob_thresholds, query=None):
        """
        Calculates a ROC curve at a specified intensity threshold.

        Args:
            model_type: type of model being evaluated (e.g. size).
            model_name: machine learning model being evaluated
            intensity_threshold: forecast bin used as the split point for evaluation
            prob_thresholds: Array of probability thresholds being evaluated.
            query: str to filter forecasts based on values of forecasts, obs, and metadata.

        Returns:
             A DistributedROC object
        """
        roc_obj = DistributedROC(prob_thresholds, 0.5)
        if query is not None:
            sub_forecasts = self.matched_forecasts[model_type][model_name].query(query)
            sub_forecasts = sub_forecasts.reset_index(drop=True)
        else:
            sub_forecasts = self.matched_forecasts[model_type][model_name]
        obs_values = np.zeros(sub_forecasts.shape[0])
        if sub_forecasts.shape[0] > 0:
            if model_type == "dist":
                forecast_values = np.array([gamma_sf(intensity_threshold, *params)
                                            for params in sub_forecasts[self.forecast_bins[model_type]].values])
                obs_probs = np.array([gamma_sf(intensity_threshold, *params)
                                    for params in sub_forecasts[self.type_cols[model_type]].values])
                obs_values[obs_probs >= 0.01] = 1
            elif len(self.forecast_bins[model_type]) > 1:
                fbin = np.argmin(np.abs(self.forecast_bins[model_type] - intensity_threshold))
                forecast_values = 1 - sub_forecasts[self.forecast_bins[model_type].astype(str)].values.cumsum(axis=1)[:,
                                    fbin]
                obs_values[sub_forecasts[self.type_cols[model_type]].values >= intensity_threshold] = 1
            else:
                forecast_values = sub_forecasts[self.forecast_bins[model_type].astype(str)[0]].values
                obs_values[sub_forecasts[self.type_cols[model_type]].values >= intensity_threshold] = 1
            roc_obj.update(forecast_values, obs_values)
        return roc_obj

    def reliability(self, model_type, model_name, intensity_threshold, prob_thresholds, query=None):
        """
        Calculate reliability statistics based on the probability of exceeding a specified threshold.

        Args:
            model_type: type of model being evaluated.
            model_name: Name of the machine learning model being evaluated.
            intensity_threshold: forecast bin used as the split point for evaluation.
            prob_thresholds: Array of probability thresholds being evaluated.
            query: str to filter forecasts based on values of forecasts, obs, and metadata.

        Returns:
            A DistributedReliability object.
        """
        rel_obj = DistributedReliability(prob_thresholds, 0.5)
        if query is not None:
            sub_forecasts = self.matched_forecasts[model_type][model_name].query(query)
            sub_forecasts = sub_forecasts.reset_index(drop=True)
        else:
            sub_forecasts = self.matched_forecasts[model_type][model_name]
        obs_values = np.zeros(sub_forecasts.shape[0])
        if sub_forecasts.shape[0] > 0:
            if model_type == "dist":
                forecast_values = np.array([gamma_sf(intensity_threshold, *params)
                                            for params in sub_forecasts[self.forecast_bins[model_type]].values])
                obs_probs = np.array([gamma_sf(intensity_threshold, *params)
                                    for params in sub_forecasts[self.type_cols[model_type]].values])
                obs_values[obs_probs >= 0.01] = 1
            elif len(self.forecast_bins[model_type]) > 1:
                fbin = np.argmin(np.abs(self.forecast_bins[model_type] - intensity_threshold))
                forecast_values = 1 - sub_forecasts[self.forecast_bins[model_type].astype(str)].values.cumsum(axis=1)[:,
                                    fbin]
                obs_values[sub_forecasts[self.type_cols[model_type]].values >= intensity_threshold] = 1
            else:
                forecast_values = sub_forecasts[self.forecast_bins[model_type].astype(str)[0]].values
                obs_values[sub_forecasts[self.type_cols[model_type]].values >= intensity_threshold] = 1
            rel_obj.update(forecast_values, obs_values)
        return rel_obj

    def sample_forecast_max_hail(self, dist_model_name, condition_model_name,
                                 num_samples, condition_threshold=0.5, query=None):
        """
        Samples every forecast hail object and returns an empirical distribution of possible maximum hail sizes.

        Hail sizes are sampled from each predicted gamma distribution. The total number of samples equals
        num_samples * area of the hail object. To get the maximum hail size for each realization, the maximum
        value within each area sample is used.

        Args:
            dist_model_name: Name of the distribution machine learning model being evaluated
            condition_model_name: Name of the hail/no-hail model being evaluated
            num_samples: Number of maximum hail samples to draw
            condition_threshold: Threshold for drawing hail samples
            query: A str that selects a subset of the data for evaluation

        Returns:
            A numpy array containing maximum hail samples for each forecast object.
        """
        if query is not None:
            dist_forecasts = self.matched_forecasts["dist"][dist_model_name].query(query)
            dist_forecasts = dist_forecasts.reset_index(drop=True)
            condition_forecasts = self.matched_forecasts["condition"][condition_model_name].query(query)
            condition_forecasts = condition_forecasts.reset_index(drop=True)
        else:
            dist_forecasts = self.matched_forecasts["dist"][dist_model_name]
            condition_forecasts = self.matched_forecasts["condition"][condition_model_name]
        max_hail_samples = np.zeros((dist_forecasts.shape[0], num_samples))
        areas = dist_forecasts["Area"].values
        for f in np.arange(dist_forecasts.shape[0]):
            condition_prob = condition_forecasts.loc[f, self.forecast_bins["condition"][0]]
            if condition_prob >= condition_threshold:
                max_hail_samples[f] = np.sort(gamma.rvs(*dist_forecasts.loc[f, self.forecast_bins["dist"]].values,
                                                        size=(num_samples, areas[f])).max(axis=1))
        return max_hail_samples

    def sample_obs_max_hail(self, dist_model_name, num_samples, query=None):
        if query is not None:
            dist_obs = self.matched_forecasts["dist"][dist_model_name].query(query)
            dist_obs = dist_obs.reset_index(drop=True)
        else:
            dist_obs = self.matched_forecasts["dist"][dist_model_name]
        max_hail_samples = np.zeros((dist_obs.shape[0], num_samples))
        areas = dist_obs["Area"].values
        for f in np.arange(dist_obs.shape[0]):
            dist_params = dist_obs.loc[f, self.type_cols["dist"]].values
            if dist_params[0] > 0:
                max_hail_samples[f] = np.sort(gamma.rvs(*dist_params,
                                                        size=(num_samples, areas[f])).max(axis=1))
        return max_hail_samples

    def max_hail_sample_crps(self, forecast_max_hail, obs_max_hail):
        crps = DistributedCRPS(thresholds=self.dist_thresholds)
        if forecast_max_hail.shape[0] > 0:
            forecast_cdfs = np.array([np.searchsorted(fs, self.dist_thresholds, side="right")
                                      for fs in forecast_max_hail]) / float(forecast_max_hail.shape[1])
            obs_cdfs = np.array([np.searchsorted(obs, self.dist_thresholds, side="right")
                                 for obs in obs_max_hail]) / float(obs_max_hail.shape[1])
            crps.update(forecast_cdfs, obs_cdfs)
        return crps




def gamma_sf(x, a, loc, b):
    if a == 0 or b == 0:
        sf = 0
    else:
        sf = gamma.sf(x, a, loc, b)
    return sf
