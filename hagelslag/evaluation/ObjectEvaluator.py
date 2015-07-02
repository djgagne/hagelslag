import numpy as np
import pandas as pd
import json
from glob import glob
from ProbabilityMetrics import DistributedCRPS, DistributedReliability, DistributedROC

class ObjectEvaluator(object):
    def __init__(self, run_date, ensemble_name, ensemble_member, model_names, model_types, forecast_bins,
                 forecast_json_path, track_data_csv_path):
        self.run_date = run_date
        self.ensemble_name = ensemble_name
        self.ensemble_member = ensemble_member
        self.model_names = model_names
        self.model_types = model_types
        self.forecast_bins = forecast_bins
        self.forecast_json_path = forecast_json_path
        self.track_data_csv_path = track_data_csv_path
        self.metadata_columns = ["Track_ID", "Obs_Track_ID", "Ensemble_Name", "Ensemble_Member", "Forecast_Hour",
                                 "Step_Duration", "Total_Duration", "Area"]
        self.type_cols = {"translation-x": "Translation_Error_X",
                          "translation-y": "Translation_Error_Y",
                          "start-time": "Start_Time_Error"}
        self.forecasts = {}
        self.obs = None
        self.matched_forecasts = {}
        for model_type in self.model_types:
            self.forecasts[model_type] = {}
            for model_name in self.model_names:
                self.forecasts[model_type][model_name] = pd.DataFrame(columns=self.metadata_columns +
                                                                      list(self.forecast_bins[model_type]))

    def load_forecasts(self):
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
                    for model_name in self.model_names:
                        prediction = feature['properties'][model_type + "_" + model_name.replace(" ", "-")]
                        if model_type == "condition":
                            prediction = [prediction]
                        row = [track_id, obs_track_id, self.ensemble_name, self.ensemble_member, forecast_hours[f],
                               f + 1, duration, area] + prediction
                        self.forecasts[model_type][model_name].ix[step_id] = row

    def load_obs(self):
        track_total_file = self.track_data_csv_path + \
            "track_total_{0}_{1}_{2}.csv".format(self.ensemble_name,
                                                 self.ensemble_member,
                                                 self.run_date.strftime("%Y%m%d"))
        track_step_file = self.track_data_csv_path + \
            "track_step__{0}_{1}_{2}.csv".format(self.ensemble_name,
                                                 self.ensemble_member,
                                                 self.run_date.strftime("%Y%m%d"))
        track_total_cols = ["Track_ID", "Translation_Error_X", "Translation_Error_Y", "Start_Time_Error"]
        track_step_cols = ["Step_ID", "Track_ID", "Hail_Size"]
        track_total_data = pd.read_csv(track_total_file, usecols=track_total_cols)
        track_step_data = pd.read_csv(track_step_file, usecols=track_step_cols)
        obs_data = pd.merge(track_step_data, track_total_data, on="Track_ID", how="left")
        self.obs = obs_data

    def merge_obs(self):
        for model_type in self.model_types:
            self.matched_forecasts[model_type] = {}
            for model_name in self.model_names:
                self.matched_forecasts[model_type][model_name] = pd.merge(self.forecasts[model_type][model_name],
                                                                          self.obs, right_on="Step_ID", how="left")

    def crps(self, model_type, model_name, query=None):
        crps_obj = DistributedCRPS(self.forecast_bins[model_type])
        if query is not None:
            sub_forecasts = self.matched_forecasts[model_type][model_name].query(query)
            crps_obj.update(sub_forecasts[self.forecast_bins[model_type]].values,
                            sub_forecasts[self.type_cols[model_type]].values)
        else:
            crps_obj.update(self.matched_forecasts[model_type][model_name][self.forecast_bins[model_type]].values,
                            self.matched_forecasts[model_type][model_name][self.type_cols[model_type]].values)
        return crps_obj
