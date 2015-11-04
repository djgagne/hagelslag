from hagelslag.data.ModelOutput import ModelOutput
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.stats import gamma
from skimage.morphology import disk
from netCDF4 import Dataset, date2num
import os
from glob import glob
import json
from datetime import timedelta


class EnsembleProducts(object):
    def __init__(self, ensemble_name, members, run_date, variable, start_date, end_date, path, single_step):
        self.ensemble_name = ensemble_name
        self.members = members
        self.run_date = run_date
        self.variable = variable
        self.start_date = start_date
        self.end_date = end_date
        self.times = pd.DatetimeIndex(start=self.start_date, end=self.end_date, freq="1H")
        self.path = path
        self.single_step = single_step
        self.data = np.array([])
        self.units = ""

    def load_data(self):
        data = []
        for member in self.members:
            mo = ModelOutput(self.ensemble_name, member, self.run_date, self.variable,
                             self.start_date, self.end_date, self.path, self.single_step)
            mo.load_data()
            if mo.units == "m":
                data.append(mo.data * 1000)
                self.units = "mm"
            else:
                data.append(mo.data)
            if self.units == "":
                self.units = mo.units
            del mo
        self.data = np.array(data)

    def point_consensus(self, consensus_type):
        if "mean" in consensus_type:
            consensus_data = np.mean(self.data, axis=0)
        elif "std" in consensus_type:
            consensus_data = np.std(self.data, axis=0)
        elif "median" in consensus_type:
            consensus_data = np.median(self.data, axis=0)
        elif "max" in consensus_type:
            consensus_data = np.max(self.data, axis=0)
        elif "percentile" in consensus_type:
            percentile = int(consensus_type.split("_")[1])
            consensus_data = np.percentile(self.data, percentile, axis=0)
        else:
            consensus_data = np.zeros(self.data.shape[1:])
        consensus = EnsembleConsensus(consensus_data, consensus_type, self.ensemble_name,
                                      self.run_date, self.variable, self.start_date, self.end_date, self.units)
        return consensus

    def point_probability(self, threshold):
        point_prob = np.zeros(self.data.shape[1:])
        for t in range(self.data.shape[1]):
            point_prob[t] = np.where(self.data[:, t] >= threshold, 1.0, 0.0).mean(axis=0)
        return EnsembleConsensus(point_prob, "point_probability", self.ensemble_name,
                                 self.run_date, self.variable + "_{0:0.2f}_{1}".format(threshold, self.units.replace(" ", "_")),
                                 self.start_date, self.end_date, "")

    def neighborhood_probability(self, threshold, radius, sigmas=None):
        if sigmas is None:
            sigmas = [0]
        weights = disk(radius)
        neighborhood_prob = np.zeros(self.data.shape[1:])
        filtered_prob = []
        for sigma in sigmas:
            filtered_prob.append(EnsembleConsensus(np.zeros(self.data.shape[1:]),
                                                   "neighbor_prob_r_{0:d}_s_{1:d}".format(radius, sigma),
                                                   self.ensemble_name,
                                                   self.run_date, self.variable + "_{0:0.2f}".format(threshold),
                                                   self.start_date, self.end_date, ""))
        thresh_data = np.where(self.data >= threshold, 1, 0)
        for t in range(self.data.shape[1]):
            for m in range(self.data.shape[0]):
                maximized = fftconvolve(thresh_data[m, t], weights, mode="same")
                maximized[maximized > 1] = 1
                neighborhood_prob[t] += fftconvolve(maximized, weights, mode="same")
            neighborhood_prob[t] /= (self.data.shape[0] * float(weights.sum()))
            for s, sigma in enumerate(sigmas):
                if sigma > 0:
                    filtered_prob[s].data[t] = gaussian_filter(neighborhood_prob[t], sigma=sigma)
                else:
                    filtered_prob[s].data[t] = neighborhood_prob[t]
        return filtered_prob

    def period_max_neighborhood_probability(self, threshold, radius, sigmas=None):
        if sigmas is None:
            sigmas = [0]
        weights = disk(radius)
        neighborhood_prob = np.zeros(self.data.shape[2:])
        for m in range(self.data.shape[0]):
            maximized = fftconvolve(np.where(self.data[m].max(axis=0) >= threshold, 1, 0), weights, mode="same")
            maximized[maximized > 1] = 1
            neighborhood_prob += fftconvolve(maximized, weights, mode="same")
        neighborhood_prob /= (self.data.shape[0] * float(weights.sum()))
        consensus_probs = []
        for sigma in sigmas:
            if sigma > 0:
                filtered_prob = gaussian_filter(neighborhood_prob, sigma=sigma)
            else:
                filtered_prob = neighborhood_prob
            ec = EnsembleConsensus(filtered_prob,
                                   "neighbor_prob_{0:02d}-hour_r_{1:d}_s_{2:d}".format(self.data.shape[1],
                                                                                       radius, sigma),
                                   self.ensemble_name,
                                   self.run_date, self.variable + "_{0:0.2f}".format(float(threshold)),
                                   self.start_date, self.end_date, "")
            consensus_probs.append(ec)
        return consensus_probs


class MachineLearningEnsembleProducts(EnsembleProducts):
    def __init__(self, ml_model_name, members, run_date, variable, start_date, end_date, grid_shape, forecast_bins,
                 forecast_json_path, condition_model_name=None):
        self.track_forecasts = {}
        self.grid_shape = grid_shape
        self.forecast_bins = forecast_bins
        self.condition_model_name = condition_model_name
        super(MachineLearningEnsembleProducts, self).__init__(ml_model_name, members, run_date, variable,
                                                              start_date, end_date, forecast_json_path,
                                                              single_step=False)

    def load_track_forecasts(self):
        run_date_str = self.run_date.strftime("%Y%m%d")
        print("Load track forecasts {0} {1}".format(self.ensemble_name, run_date_str))
        for member in self.members:
            self.track_forecasts[member] = []
            track_files = sorted(glob(self.path + "/".join([run_date_str, member]) + "/*.json"))
            if len(track_files) > 0:
                self.track_forecasts[member] = []
                for track_file in track_files:
                    tfo = open(track_file)
                    self.track_forecasts[member].append(json.load(tfo))
                    tfo.close()
                    del tfo
        return

    def load_data(self, grid_method="mean"):
        if self.track_forecasts == {}:
            self.load_track_forecasts()
        if self.track_forecasts == {}:
            return -1
        self.data = np.zeros((len(self.members), self.times.size, self.grid_shape[0], self.grid_shape[1]))
        if grid_method in ["mean", "median", "samples"]:
            for m, member in enumerate(self.members):
                for track_forecast in self.track_forecasts[member]:
                    times = track_forecast["properties"]["times"]
                    for s, step in enumerate(track_forecast["features"]):
                        forecast_pdf = np.array(step['properties'][self.variable + "_" +
                                                                   self.ensemble_name.replace(" ", "-")])
                        forecast_time = self.run_date + timedelta(hours=times[s])
                        t = np.where(self.times == forecast_time)[0][0]
                        mask = np.array(step['properties']["masks"], dtype=int)
                        i = np.array(step['properties']["i"], dtype=int)
                        i = i[mask == 1]
                        j = np.array(step['properties']["j"], dtype=int)
                        j = j[mask == 1]
                        if grid_method == "samples":
                            intensities = np.array(step["properties"]["timesteps"], dtype=float)[mask == 1]
                            rankings = np.argsort(intensities)
                            samples = np.random.choice(self.forecast_bins, size=intensities.size, replace=True,
                                                       p=forecast_pdf)
                            self.data[m, t, i[rankings], j[rankings]] = samples
                        else:
                            if grid_method == "mean":
                                forecast_value = np.sum(forecast_pdf * self.forecast_bins)
                            elif grid_method == "median":
                                forecast_cdf = np.cumsum(forecast_pdf)
                                forecast_value = self.forecast_bins[np.argmin(np.abs(forecast_cdf - 0.5))]
                            else:
                                forecast_value = 0
                            self.data[m, t, i, j] = forecast_value
        if grid_method in ["gamma"]:
            for m, member in enumerate(self.members):
                for track_forecast in self.track_forecasts[member]:
                    times = track_forecast["properties"]["times"]
                    for s, step in enumerate(track_forecast["features"]):
                        forecast_params = step["properties"][self.variable + "_" + self.ensemble_name.replace(" ", "-")]
                        if self.condition_model_name is not None:
                            condition = step["properties"]["condition_" + self.condition_model_name.replace(" ", "-")]
                        else:
                            condition = None
                        forecast_dist = gamma(forecast_params[0], loc=forecast_params[1], scale=forecast_params[2])
                        forecast_time = self.run_date + timedelta(hours=times[s])
                        if forecast_time in self.times:
                            t = np.where(self.times == forecast_time)[0][0]
                            mask = np.array(step["properties"]["masks"], dtype=int)
                            rankings = np.argsort(step["properties"]["timesteps"])[mask == 1]
                            i = np.array(step["properties"]["i"], dtype=int)[mask == 1][rankings]
                            j = np.array(step["properties"]["j"], dtype=int)[mask == 1][rankings]
                            if rankings.size > 0:
                                samples = np.sort(forecast_dist.rvs(size=rankings.size))
                                if condition is None or condition >= 0.5:
                                    self.data[m, t, i, j] = samples
        return 0


class EnsembleConsensus(object):
    """
    Stores data and metadata for an ensemble consensus product such as a neighborhood probability or an ensemble
    mean or max. Allows for the product to be output to a netCDF file.
    """
    def __init__(self, data, consensus_type, ensemble_name, run_date, variable, start_date, end_date, units):
        self.data = data
        self.data[self.data < 0.001] = 0
        self.consensus_type = consensus_type
        self.ensemble_name = ensemble_name
        self.run_date = run_date
        self.variable = variable
        self.start_date = start_date
        self.end_date = end_date
        self.times = pd.DatetimeIndex(start=start_date, end=end_date, freq="1H")
        self.units = units

    def to_file(self, filename, time_units="seconds since 1970-01-01T00:00"):
        """
        Outputs data to a netCDF file. If the file does not exist, it will be created. Otherwise, additional variables
        are appended to the current file

        :param filename: Full-path and name of output netCDF file
        :param time_units: Units for the time variable in format "<time> since <date string>"
        :return:
        """
        full_var_name = self.consensus_type + "_" + self.variable 
        if os.access(filename, os.R_OK):
            out_data = Dataset(filename, "r+")
        else:
            out_data = Dataset(filename, "w")
            for d, dim in enumerate(["time", "y", "x"]):
                out_data.createDimension(dim, self.data.shape[d])
            time_var = out_data.createVariable("time", "i8", ("time",))
            time_var[:] = date2num(self.times.to_pydatetime(), time_units)
            time_var.units = time_units
        if "-hour" in self.consensus_type:
            if full_var_name not in out_data.variables.keys():
                var = out_data.createVariable(full_var_name, "f4", ("y", "x"), zlib=True, least_significant_digit=4)
            else:
                var = out_data.variables[full_var_name]
            var.coordinates = "y x"
        else:
            if full_var_name not in out_data.variables.keys():
                var = out_data.createVariable(full_var_name, "f4", ("time", "y", "x"), zlib=True, least_significant_digit=4)
            else:
                var = out_data.variables[full_var_name]
            var.coordinates = "time y x"
        var[:] = self.data
        var.units = self.units
        var.long_name = self.consensus_type + "_" + self.variable
        out_data.Conventions = "CF-1.6"
        out_data.close()
        return
