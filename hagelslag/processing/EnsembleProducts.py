from hagelslag.data.ModelOutput import ModelOutput
from hagelslag.util.make_proj_grids import read_arps_map_file, read_ncar_map_file, make_proj_grids
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.stats import gamma, bernoulli
from skimage.morphology import disk
from netCDF4 import Dataset, date2num
import os
from glob import glob
import json
from datetime import timedelta
from multiprocessing import Queue, Pool
import traceback

try:
    from ncepgrib2 import Grib2Encode, dump
    grib_support = True
except ImportError("ncepgrib2 not available"):
    grib_support = False




class EnsembleMemberProduct(object):
    def __init__(self, ensemble_name, model_name, member, run_date, variable, start_date, end_date, path, single_step,
                 map_file=None, condition_model_name=None, condition_threshold=0.5):
        self.ensemble_name = ensemble_name
        self.model_name = model_name
        self.member = member
        self.run_date = run_date
        self.variable = variable
        self.start_date = start_date
        self.end_date = end_date
        self.times = pd.DatetimeIndex(start=self.start_date, end=self.end_date, freq="1H")
        self.forecast_hours = (self.times - self.run_date).astype('timedelta64[h]').values
        self.path = path
        self.single_step = single_step
        self.track_forecasts = None
        self.data = None
        self.map_file = map_file
        self.proj_dict = None
        self.grid_dict = None
        self.mapping_data = None
        self.condition_model_name = condition_model_name
        self.condition_threshold = condition_threshold
        self.percentiles = None
        self.num_samples = None
        self.percentile_data = None
        if self.map_file is not None:
            if self.map_file[-3:] == "map":
                self.proj_dict, self.grid_dict = read_arps_map_file(self.map_file)
            else:
                self.proj_dict, self.grid_dict = read_ncar_map_file(self.map_file)
            self.mapping_data = make_proj_grids(self.proj_dict, self.grid_dict)
        self.units = ""

    def load_data(self, num_samples=1000, percentiles=None):
        """
        Args:
            num_samples:
            percentiles:

        Returns:
        """
        self.percentiles = percentiles
        self.num_samples = num_samples
        if self.model_name.lower() in ["wrf"]:
            mo = ModelOutput(self.ensemble_name, self.member, self.run_date, self.variable,
                             self.start_date, self.end_date, self.path, self.single_step)
            mo.load_data()
            self.data = mo.data[:]
            if mo.units == "m":
                self.data *= 1000
                self.units = "mm"
            else:
                self.units = mo.units
        else:
            if self.track_forecasts is None:
                self.load_track_data()
            self.units = "mm"
            self.data = np.zeros((self.forecast_hours.size,
                                  self.mapping_data["lon"].shape[0],
                                  self.mapping_data["lon"].shape[1]), dtype=np.float32)
            
            if self.percentiles is not None:
                self.percentile_data = np.zeros([len(self.percentiles)] + list(self.data.shape))
            full_condition_name = "condition_" + self.condition_model_name.replace(" ", "-")
            dist_model_name = "dist" + "_" + self.model_name.replace(" ", "-")
            for track_forecast in self.track_forecasts:
                times = track_forecast["properties"]["times"]
                for s, step in enumerate(track_forecast["features"]):
                    forecast_params = step["properties"][dist_model_name]
                    if self.condition_model_name is not None:
                        condition = step["properties"][full_condition_name]
                    else:
                        condition = None
                    forecast_time = self.run_date + timedelta(hours=times[s])
                    if forecast_time in self.times:
                        t = np.where(self.times == forecast_time)[0][0]
                        mask = np.array(step["properties"]["masks"], dtype=int)
                        rankings = np.argsort(step["properties"]["timesteps"])[mask == 1]
                        i = np.array(step["properties"]["i"], dtype=int)[mask == 1][rankings]
                        j = np.array(step["properties"]["j"], dtype=int)[mask == 1][rankings]
                        if rankings.size > 0:
                            raw_samples = np.sort(gamma.rvs(forecast_params[0], loc=forecast_params[1],
                                                            scale=forecast_params[2],
                                                            size=(num_samples, rankings.size)),
                                                  axis=1)
                            #raw_samples *= bernoulli.rvs(condition,
                            #                             size=(num_samples, rankings.size))
                            if self.percentiles is None:
                                samples = raw_samples.mean(axis=0)
                                if condition is None or condition >= self.condition_threshold:
                                    self.data[t, i, j] = samples
                            else:
                                for p, percentile in enumerate(self.percentiles):
                                    if percentile != "mean":
                                        self.percentile_data[p, t, i, j] = np.percentile(raw_samples, percentile, axis=0)
                                    else:
                                        self.percentile_data[p, t, i, j] = np.mean(raw_samples, axis=0)
                                samples = raw_samples.mean(axis=0)
                                if condition is None or condition >= self.condition_threshold:
                                    self.data[t, i, j] = samples

    def load_track_data(self):
        run_date_str = self.run_date.strftime("%Y%m%d")
        print("Load track forecasts {0} {1}".format(self.ensemble_name, run_date_str))
        track_files = sorted(glob(self.path + "/".join([run_date_str, self.member]) + "/*.json"))
        if len(track_files) > 0:
            self.track_forecasts = []
            for track_file in track_files:
                tfo = open(track_file)
                self.track_forecasts.append(json.load(tfo))
                tfo.close()
        else:
            self.track_forecasts = []
    def neighborhood_probability(self, threshold, radius):
        weights = disk(radius, dtype=np.uint8)
        thresh_data = np.zeros(self.data.shape[1:], dtype=np.uint8)
        neighbor_prob = np.zeros(self.data.shape, dtype=np.float32)
        for t in np.arange(self.data.shape[0]):
            thresh_data[self.data[t] >= threshold] = 1
            maximized = fftconvolve(thresh_data, weights, mode="same")
            maximized[maximized > 1] = 1
            maximized[maximized < 1] = 0
            neighbor_prob[t] = fftconvolve(maximized, weights, mode="same")
            thresh_data[:] = 0
        neighbor_prob[neighbor_prob < 1] = 0
        neighbor_prob /= weights.sum()
        return neighbor_prob

    def period_max_neighborhood_probability(self, threshold, radius):
        weights = disk(radius, dtype=np.uint8)
        thresh_data = np.zeros(self.data.shape[1:], dtype=np.uint8)
        thresh_data[self.data.max(axis=0) >= threshold] = 1
        maximized = fftconvolve(thresh_data, weights, mode="same")
        maximized[maximized > 1] = 1
        maximized[maximized < 1] = 0
        neighborhood_prob = fftconvolve(maximized, weights, mode="same")
        neighborhood_prob[neighborhood_prob < 1] = 0
        neighborhood_prob /= weights.sum()
        return neighborhood_prob

    def encode_grib2(self):
        """
        Encodes member percentile data to GRIB2 format.

        Returns:
            Series of
        """
        lscale = 1e6
        grib_id_start = [7, 0, 14, 14, 2]
        gdsinfo = np.array([0, np.product(self.data.shape[-2:]), 0, 0, 30], dtype=np.int32)
        lon_0 = self.proj_dict["lon_0"]
        sw_lon = self.grid_dict["sw_lon"]
        if lon_0 < 0:
            lon_0 += 360
        if sw_lon < 0:
            sw_lon += 360
        gdtmp1 = [1, 0, self.proj_dict['a'], 3, float(self.proj_dict['a']), 3, float(self.proj_dict['b']),
                           self.data.shape[-1], self.data.shape[-2], self.grid_dict["sw_lat"] * lscale,
                           sw_lon * lscale, 0, self.proj_dict["lat_0"] * lscale,
                           lon_0 * lscale,
                           self.grid_dict["dx"] * 1e3, self.grid_dict["dy"] * 1e3, 0b00000000, 0b01000000,
                           self.proj_dict["lat_1"] * lscale,
                           self.proj_dict["lat_2"] * lscale, -90 * lscale, 0]
        pdtmp1 = np.array([1,                # parameter category Moisture
                           31,               # parameter number Hail
                           4,                # Type of generating process Ensemble Forecast
                           0,                # Background generating process identifier
                           31,               # Generating process or model from NCEP
                           0,                # Hours after reference time data cutoff
                           0,                # Minutes after reference time data cutoff
                           1,                # Forecast time units Hours
                           0,                # Forecast time
                           1,                # Type of first fixed surface Ground
                           1,                # Scale value of first fixed surface
                           0,                # Value of first fixed surface
                           1,                # Type of second fixed surface
                           1,                # Scale value of 2nd fixed surface
                           0,                # Value of 2nd fixed surface
                           0,                # Derived forecast type
                           self.num_samples  # Number of ensemble members
                           ], dtype=np.int32)
        #pdtmp1 = np.array([1, 31, 2, 0, 116, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 192, 0, self.data.shape[0]], dtype=np.int32)
        grib_objects = pd.Series(index=self.times, data=[None] * self.times.size, dtype=object)
        drtmp1 = np.array([0, 0, 4, 8, 0], dtype=np.int32)
        for t, time in enumerate(self.times):
            time_list = list(time.utctimetuple()[0:6])
            if grib_objects[time] is None:
                grib_objects[time] = Grib2Encode(0, np.array(grib_id_start + time_list + [2, 1], dtype=np.int32))
                grib_objects[time].addgrid(gdsinfo, gdtmp1)
            pdtmp1[8] = (time.to_pydatetime() - self.run_date).total_seconds() / 3600.0
            data = self.percentile_data[:, t] / 1000.0
            masked_data = np.ma.array(data, mask=data <= 0)
            for p, percentile in enumerate(self.percentiles):
                print("GRIB {3} Percentile {0}. Max: {1} Min: {2}".format(percentile, 
                                                                          masked_data[p].max(), 
                                                                          masked_data[p].min(),
                                                                          time))
                if percentile in range(1, 100):
                    pdtmp1[-2] = percentile
                    grib_objects[time].addfield(6, pdtmp1[:-1], 0, drtmp1, masked_data[p])
                else:
                    pdtmp1[-2] = 0
                    grib_objects[time].addfield(2, pdtmp1, 0, drtmp1, masked_data[p])
        return grib_objects

    def write_grib2_files(self, grib_objects, path):
        for time in self.times.to_pydatetime():
            grib_objects[time].end()
            filename = path  + "{0}_{1}_{2}_{3}_{4}.grib2".format(self.ensemble_name,
                                                                  self.member,
                                                                  self.model_name.replace(" ", "-"),
                                                                  self.variable,
                                                                  time.strftime("%Y%m%d%H%M")
                                                                  )
            fo = open(filename, "wb")
            fo.write(grib_objects[time].msg)
            fo.close()

class EnsembleProducts(object):
    """
    Loads in individual ensemble members and generates both grid point and neighborhood probabilities from the
    model output.

    Attributes:
        ensemble_name (str): Name of the ensemble prediction system.
        members (list): List of ensemble member names.
        run_date (datetime.datetime): Initial date and time of the model run
        variable (str): Name of model variable being processed
        start_date (datetime.datetime or date string): Date of the initial forecast time step.
        end_date (datetime.datetime or date string): Date of the final forecast time step.
        times (pandas.DatetimeIndex): A sequence of hourly forecast times.
        path (str): Path to the ensemble model files.
        single_step (bool): If True, each forecast timestep is in a separate file.
        data (ndarray[members, times, y, x]): Forecast data. Initially None.
        units (str): The units of the forecast variable
    """
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
        self.data = None
        self.units = ""
    
    def load_data(self):
        """
        Loads data from each ensemble member.
        """
        for m, member in enumerate(self.members):
            mo = ModelOutput(self.ensemble_name, member, self.run_date, self.variable,
                             self.start_date, self.end_date, self.path, self.single_step)
            mo.load_data()
            if self.data is None:
                self.data = np.zeros((len(self.members), mo.data.shape[0], mo.data.shape[1], mo.data.shape[2]),
                                     dtype=np.float32)
            if mo.units == "m":
                self.data[m] = mo.data * 1000
                self.units = "mm"
            else:
                self.data[m] = mo.data
            if self.units == "":
                self.units = mo.units
            del mo.data
            del mo

    def point_consensus(self, consensus_type):
        """
        Calculate grid-point statistics across ensemble members.

        Args:
            consensus_type: mean, std, median, max, or percentile_nn

        Returns:
            EnsembleConsensus containing point statistic
        """
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
        """
        Determine the probability of exceeding a threshold at a grid point based on the ensemble forecasts at
        that point.

        Args:
            threshold: If >= threshold assigns a 1 to member, otherwise 0.

        Returns:
            EnsembleConsensus
        """
        point_prob = np.zeros(self.data.shape[1:])
        for t in range(self.data.shape[1]):
            point_prob[t] = np.where(self.data[:, t] >= threshold, 1.0, 0.0).mean(axis=0)
        return EnsembleConsensus(point_prob, "point_probability", self.ensemble_name,
                                 self.run_date, self.variable + "_{0:0.2f}_{1}".format(threshold,
                                                                                       self.units.replace(" ", "_")),
                                 self.start_date, self.end_date, "")

    def neighborhood_probability(self, threshold, radius, sigmas=None):
        """
        Hourly probability of exceeding a threshold based on model values within a specified radius of a point.

        Args:
            threshold (float): probability of exceeding this threshold
            radius (int): distance from point in number of grid points to include in neighborhood calculation.
            sigmas (array of ints): Radii for Gaussian filter used to smooth neighborhood probabilities.

        Returns:
            list of EnsembleConsensus objects containing neighborhood probabilities for each forecast hour.
        """
        if sigmas is None:
            sigmas = [0]
        weights = disk(radius)
        filtered_prob = []
        for sigma in sigmas:
            filtered_prob.append(EnsembleConsensus(np.zeros(self.data.shape[1:], dtype=np.float32),
                                                   "neighbor_prob_r_{0:d}_s_{1:d}".format(radius, sigma),
                                                   self.ensemble_name,
                                                   self.run_date, self.variable + "_{0:0.2f}".format(threshold),
                                                   self.start_date, self.end_date, ""))
        thresh_data = np.zeros(self.data.shape[2:], dtype=np.uint8)
        neighbor_prob = np.zeros(self.data.shape[2:], dtype=np.float32)
        for t in range(self.data.shape[1]):
            for m in range(self.data.shape[0]):
                thresh_data[self.data[m, t] >= threshold] = 1
                maximized = fftconvolve(thresh_data, weights, mode="same")
                maximized[maximized > 1] = 1
                maximized[maximized < 1] = 0
                neighbor_prob += fftconvolve(maximized, weights, mode="same")
                neighbor_prob[neighbor_prob < 1] = 0
                thresh_data[:] = 0
            neighbor_prob /= (self.data.shape[0] * float(weights.sum()))
            for s, sigma in enumerate(sigmas):
                if sigma > 0:
                    filtered_prob[s].data[t] = gaussian_filter(neighbor_prob, sigma=sigma)
                else:
                    filtered_prob[s].data[t] = neighbor_prob
            neighbor_prob[:] = 0
        return filtered_prob

    def period_max_neighborhood_probability(self, threshold, radius, sigmas=None):
        """
        Calculates the neighborhood probability of exceeding a threshold at any time over the period loaded.

        Args:
            threshold (float): splitting threshold for probability calculatations
            radius (int): distance from point in number of grid points to include in neighborhood calculation.
            sigmas (array of ints): Radii for Gaussian filter used to smooth neighborhood probabilities.

        Returns:
            list of EnsembleConsensus objects
        """
        if sigmas is None:
            sigmas = [0]
        weights = disk(radius)
        neighborhood_prob = np.zeros(self.data.shape[2:], dtype=np.float32)
        thresh_data = np.zeros(self.data.shape[2:], dtype=np.uint8)
        for m in range(self.data.shape[0]):
            thresh_data[self.data[m].max(axis=0) >= threshold] = 1
            maximized = fftconvolve(thresh_data, weights, mode="same")
            maximized[maximized > 1] = 1
            neighborhood_prob += fftconvolve(maximized, weights, mode="same")
        neighborhood_prob[neighborhood_prob < 1] = 0
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
    """
    Subclass of EnsembleProducts that processes forecasts from machine learning models. In particular, this
    class converts object distribution forecasts to grid point values for each ensemble member.
    """
    def __init__(self, ml_model_name, members, run_date, variable, start_date, end_date, grid_shape, forecast_bins,
                 forecast_json_path, condition_model_name=None, map_file=None):
        self.track_forecasts = {}
        self.grid_shape = grid_shape
        self.forecast_bins = forecast_bins
        self.condition_model_name = condition_model_name
        self.percentile = None
        if map_file is not None:
            if map_file[-3:] == "map":
                self.proj_dict, self.grid_dict = read_arps_map_file(map_file)
            else:
                self.proj_dict, self.grid_dict = read_ncar_map_file(map_file)
        else:
            self.proj_dict = None
            self.grid_dict = None
        super(MachineLearningEnsembleProducts, self).__init__(ml_model_name, members, run_date, variable,
                                                              start_date, end_date, forecast_json_path,
                                                              single_step=False)

    def load_track_forecasts(self):
        """
        Load the track forecasts from each geoJSON file.

        """
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
    
    def load_data(self, grid_method="gamma", num_samples=1000, condition_threshold=0.5, zero_inflate=False,
                  percentile=None):
        """
        Reads the track forecasts and converts them to grid point values based on random sampling.

        Args:
            grid_method: "gamma" by default
            num_samples: Number of samples drawn from predicted pdf
            condition_threshold: Objects are not written to the grid if condition model probability is below this
                threshold.
            zero_inflate: Whether to sample zeros from a Bernoulli sampler based on the condition model probability
            percentile: If None, outputs the mean of the samples at each grid point, otherwise outputs the specified
                percentile from 0 to 100.

        Returns:
            0 if tracks are successfully sampled on to grid. If no tracks are found, returns -1.
        """
        self.percentile = percentile
        if self.track_forecasts == {}:
            self.load_track_forecasts()
        if self.track_forecasts == {}:
            return -1
        if self.data is None:
            self.data = np.zeros((len(self.members), self.times.size, self.grid_shape[0], self.grid_shape[1]),
                                 dtype=np.float32)
        else:
            self.data[:] = 0
        if grid_method in ["mean", "median", "samples"]:
            for m, member in enumerate(self.members):
                print("Sampling " + member)
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
            full_condition_name = "condition_" + self.condition_model_name.replace(" ", "-")
            dist_model_name = self.variable + "_" + self.ensemble_name.replace(" ", "-")
            for m, member in enumerate(self.members):
                for track_forecast in self.track_forecasts[member]:
                    times = track_forecast["properties"]["times"]
                    for s, step in enumerate(track_forecast["features"]):
                        forecast_params = step["properties"][dist_model_name]
                        if self.condition_model_name is not None:
                            condition = step["properties"][full_condition_name]
                        else:
                            condition = None
                        forecast_time = self.run_date + timedelta(hours=times[s])
                        if forecast_time in self.times:
                            t = np.where(self.times == forecast_time)[0][0]
                            mask = np.array(step["properties"]["masks"], dtype=int)
                            rankings = np.argsort(step["properties"]["timesteps"])[mask == 1]
                            i = np.array(step["properties"]["i"], dtype=int)[mask == 1][rankings]
                            j = np.array(step["properties"]["j"], dtype=int)[mask == 1][rankings]
                            if rankings.size > 0:
                                raw_samples = np.sort(gamma.rvs(forecast_params[0], loc=forecast_params[1],
                                                                scale=forecast_params[2],
                                                                size=(num_samples, rankings.size)),
                                                      axis=1)
                                if zero_inflate:
                                    raw_samples *= bernoulli.rvs(condition,
                                                                 size=(num_samples, rankings.size))
                                if percentile is None:
                                    samples = raw_samples.mean(axis=0)
                                else:
                                    samples = np.percentile(raw_samples, percentile, axis=0)
                                if condition is None or condition >= condition_threshold:
                                    self.data[m, t, i, j] = samples
        return 0

    def write_grib2(self, path):
        """
        Writes data to grib2 file. Currently, grib codes are set by hand to hail.

        Args:
            path: Path to directory containing grib2 files.

        Returns:

        """
        if self.percentile is None:
            var_type = "mean"
        else:
            var_type = "p{0:02d}".format(self.percentile)
        lscale = 1e6
        grib_id_start = [7, 0, 14, 14, 2]
        gdsinfo = np.array([0, np.product(self.data.shape[-2:]), 0, 0, 30], dtype=np.int32)
        lon_0 = self.proj_dict["lon_0"]
        sw_lon = self.grid_dict["sw_lon"]
        if lon_0 < 0:
            lon_0 += 360
        if sw_lon < 0:
            sw_lon += 360
        gdtmp1 = np.array([7, 1, self.proj_dict['a'], 1, self.proj_dict['a'], 1, self.proj_dict['b'],
                           self.data.shape[-2], self.data.shape[-1], self.grid_dict["sw_lat"] * lscale,
                           sw_lon * lscale, 0, self.proj_dict["lat_0"] * lscale,
                           lon_0 * lscale,
                           self.grid_dict["dx"] * 1e-3, self.grid_dict["dy"] * 1e-3, 0,
                           self.proj_dict["lat_1"] * lscale,
                           self.proj_dict["lat_2"] * lscale, 0, 0], dtype=np.int32)
        pdtmp1 = np.array([1, 31, 2, 0, 116, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 192, 0, self.data.shape[0]], dtype=np.int32)
        for m, member in enumerate(self.members):
            pdtmp1[-2] = m
            for t, time in enumerate(self.times):
                time_list = list(time.utctimetuple()[0:6])
                grbe = Grib2Encode(0, np.array(grib_id_start + time_list + [2, 1], dtype=np.int32))
                grbe.addgrid(gdsinfo, gdtmp1)
                pdtmp1[8] = (time.to_pydatetime() - self.run_date).total_seconds() / 3600.0
                drtmp1 = np.array([0, 0, 4, 8, 0], dtype=np.int32)
                data = self.data[m, t].astype(np.float32) / 1000.0
                masked_data = np.ma.array(data, mask=data<=0)
                grbe.addfield(1, pdtmp1, 0, drtmp1, masked_data)
                grbe.end()
                filename = path + "{0}_{1}_mlhail_{2}_{3}.grib2".format(self.ensemble_name.replace(" ", "-"), member, var_type,
                                                                        time.to_datetime().strftime("%Y%m%d%H%M"))
                print("Writing to " + filename)
                grib_file = open(filename, "wb")
                grib_file.write(grbe.msg)
                grib_file.close()
        return


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

    def init_file(self, filename, time_units="seconds since 1970-01-01T00:00"):
        """
        Initializes netCDF file for writing

        Args:
            filename: Name of the netCDF file
            time_units: Units for the time variable in format "<time> since <date string>"
        Returns:
            Dataset object
        """
        if os.access(filename, os.R_OK):
            out_data = Dataset(filename, "r+")
        else:
            out_data = Dataset(filename, "w")
            for d, dim in enumerate(["time", "y", "x"]):
                out_data.createDimension(dim, self.data.shape[d])

            time_var = out_data.createVariable("time", "i8", ("time",))
            time_var[:] = date2num(self.times.to_pydatetime(), time_units)
            time_var.units = time_units
            out_data.Conventions = "CF-1.6"
        return out_data

    def write_to_file(self, out_data):
        """
        Outputs data to a netCDF file. If the file does not exist, it will be created. Otherwise, additional variables
        are appended to the current file

        Args:
            out_data: Full-path and name of output netCDF file
        """
        full_var_name = self.consensus_type + "_" + self.variable
        if "-hour" in self.consensus_type:
            if full_var_name not in out_data.variables.keys():
                var = out_data.createVariable(full_var_name, "f4", ("y", "x"), zlib=True, 
                                              least_significant_digit=3, shuffle=True)
            else:
                var = out_data.variables[full_var_name]
            var.coordinates = "y x"
        else:
            if full_var_name not in out_data.variables.keys():
                var = out_data.createVariable(full_var_name, "f4", ("time", "y", "x"), zlib=True,
                                              least_significant_digit=3, shuffle=True)
            else:
                var = out_data.variables[full_var_name]
            var.coordinates = "time y x"
        var[:] = self.data
        var.units = self.units
        var.long_name = self.consensus_type + "_" + self.variable
        return

