from hagelslag.data.ModelOutput import ModelOutput
from hagelslag.util.make_proj_grids import read_arps_map_file, read_ncar_map_file, make_proj_grids
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.stats import gamma, bernoulli
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from skimage.morphology import disk
from netCDF4 import Dataset, date2num, num2date
import os
from glob import glob
from os.path import join, exists
import json
from datetime import timedelta


try:
    from ncepgrib2 import Grib2Encode, dump
    grib_support = True
except ImportError("ncepgrib2 not available"):
    grib_support = False


class EnsembleMemberProduct(object):
    """
    This class loads machine learning forecasts for a single ensemble member and run and converts them to a gridded
    field.

    Args:
        ensemble_name (str): name of the ensemble (e.g., HREF, SSEF, NCAR, etc.)
        model_name (str): name of the machine learning model
        member (str): name of the ensemble member
        run_date (`datetime.datetime`): date of the initial time step of the model run
        variable (str): name of the variable used for object-extraction
        start_date (`datetime.datetime`): Start of the model extraction period
        end_date (`datetime.datetime`): End of the model extraction period
        path (str): Path to model output
        single_step (bool): Whether or not the model output is stored in single files or multiple files.
        size_distribution_training_path (str): Path to size distribution percentiles
        watershed_var (str): Name of variable used for object extraction.
        map_file (str or None): Map projection file for given ensemble type.
        condition_model_name (str): Name of the condition ML model being used if different from model_name
        condition_threshold (float): Probability threshold for including or excluding storms.
    """
    def __init__(self, ensemble_name, model_name, member, run_date, variable, start_date, end_date, path, single_step,
                 size_distribution_training_path, watershed_var, map_file=None,
                 condition_model_name=None, condition_threshold=0.5):
        self.ensemble_name = ensemble_name
        self.model_name = model_name
        self.member = member
        self.run_date = run_date
        self.variable = variable
        self.start_date = start_date
        self.end_date = end_date
        self.times = pd.date_range(start=self.start_date, end=self.end_date, freq="1H")
        self.forecast_hours = (self.times - self.run_date).astype('timedelta64[h]').values
        self.path = path
        self.single_step = single_step
        self.size_distribution_training_path = size_distribution_training_path
        self.watershed_var = watershed_var
        self.track_forecasts = None
        self.data = None
        self.map_file = map_file
        self.proj_dict = None
        self.grid_dict = None
        self.mapping_data = None
        if condition_model_name is None:
            self.condition_model_name = model_name
        else:
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
        self.nc_patches = None
        self.hail_forecast_table = None

    def load_data(self, num_samples=1000, percentiles=None):
        """
        Load data from forecast json files and map forecasts to grid with percentile method.

        Args:
            num_samples: Number of random samples at each grid point
            percentiles: Which percentiles to extract from the random samples

        Returns:
        """
        self.percentiles = percentiles
        self.num_samples = num_samples
        if self.model_name.lower() in ["wrf"]:
            mo = ModelOutput(self.ensemble_name, self.member, self.run_date, self.variable,
                             self.start_date, self.end_date, self.path, self.map_file, self.single_step)
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
                        mask = np.array(step["properties"]["masks"], dtype=int).ravel()
                        rankings = np.argsort(np.array(step["properties"]["timesteps"]).ravel()[mask==1])
                        i = np.array(step["properties"]["i"], dtype=int).ravel()[mask == 1][rankings]
                        j = np.array(step["properties"]["j"], dtype=int).ravel()[mask == 1][rankings]
                        if rankings.size > 0 and forecast_params[0] > 0.1 and 1 < forecast_params[2] < 100:
                            raw_samples = np.sort(gamma.rvs(forecast_params[0], loc=forecast_params[1],
                                                            scale=forecast_params[2],
                                                            size=(num_samples, rankings.size)),
                                                  axis=1)
                            if self.percentiles is None:
                                samples = raw_samples.mean(axis=0)
                                if condition >= self.condition_threshold:
                                    self.data[t, i, j] = samples
                            else:
                                for p, percentile in enumerate(self.percentiles):
                                    if percentile != "mean":
                                        if condition >= self.condition_threshold:
                                            self.percentile_data[p, t, i, j] = np.percentile(raw_samples, percentile,
                                                                                             axis=0)
                                    else:
                                        if condition >= self.condition_threshold:
                                            self.percentile_data[p, t, i, j] = np.mean(raw_samples, axis=0)
                                samples = raw_samples.mean(axis=0)
                                if condition >= self.condition_threshold:
                                    self.data[t, i, j] = samples

    def load_track_data(self):
        """
        Load track forecats from json files and input info to self.track_forecasts.

        """
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

    def load_forecast_csv_data(self, csv_path):
        """
        Load track forecast csv files with pandas.

        Args:
            csv_path: Path to csv files.

        Returns:

        """
        forecast_file = join(csv_path, "hail_forecasts_{0}_{1}_{2}.csv".format(self.ensemble_name,
                                                                    self.member,
                                                                    self.run_date.strftime("%Y%m%d-%H%M")))
        if exists(forecast_file):
            self.hail_forecast_table = pd.read_csv(forecast_file)
        return

    def load_forecast_netcdf_data(self, nc_path):
        """
        Load netCDF patches for each storm.

        Args:
            nc_path: Path to forecast netCDF files.

        """
        nc_file = join(nc_path, "{0}_{1}_{2}_model_patches.nc".format(self.ensemble_name,
                                                                      self.run_date.strftime("%Y%m%d-%H%M"),
                                                                      self.member))
        if exists(nc_file):
            nc_patches = Dataset(nc_file)
            nc_times = pd.DatetimeIndex(num2date(nc_patches.variables["time"][:],
                                             nc_patches.variables["time"].units))
            time_indices = np.isin(nc_times, self.times)
            self.nc_patches = dict()
            self.nc_patches["time"] = nc_times[time_indices]
            self.nc_patches["forecast_hour"] = nc_patches.variables["time"][time_indices]
            self.nc_patches["obj_values"] = nc_patches.variables[nc_patches.object_variable + "_curr"][time_indices]
            self.nc_patches["masks"] = nc_patches.variables["masks"][time_indices]
            self.nc_patches["i"] = nc_patches.variables["i"][time_indices]
            self.nc_patches["j"] = nc_patches.variables["j"][time_indices]
            nc_patches.close()
            print(nc_file)
        else:
            print('no {0} {1} netCDF4 file'.format(self.member,self.run_date.strftime("%Y%m%d")))
            self.nc_patches = None
            return 
        return
    def quantile_match(self):
        """
        For each storm object, get the percentiles of the enhanced watershed variable field relative to the training
        climatology of that variable. Then, extract the hail sizes at those percentiles from the predicted hail size
        distribution for each storm. If the probability of hail occurring exceeds the model's condition
        threshold, then the storm is written to the data grid.

        """
        if self.nc_patches is None:
            self.data = None
            return 

        mask_indices = np.where(self.nc_patches["masks"] == 1)
        obj_values = self.nc_patches["obj_values"][mask_indices]
        obj_values = np.array(obj_values)
        percentiles = np.linspace(0.1, 99.9, 100)
        
        try:
            filename = join(self.size_distribution_training_path,
                            '{0}_{1}_{2}_Size_Distribution.csv'.format(self.ensemble_name,
                                                                       self.watershed_var,
                                                                       self.member))
            if not exists(filename):
                filename = join(self.size_distribution_training_path,
                        '{0}_{1}_Size_Distribution.csv'.format(self.ensemble_name,
                                                                self.watershed_var))
            train_period_obj_per_vals = pd.read_csv(filename)
            train_period_obj_per_vals = train_period_obj_per_vals.loc[:,"Obj_Values"].values
            per_func = interp1d(train_period_obj_per_vals, percentiles / 100.0, 
                                bounds_error=False, fill_value=(0.1, 99.9))
        except FileNotFoundError:
            obj_per_vals = np.percentile(obj_values, percentiles)
            per_func = interp1d(obj_per_vals, percentiles / 100.0, bounds_error=False, fill_value=(0.1, 99.9))

        obj_percentiles = np.zeros(self.nc_patches["masks"].shape)
        obj_percentiles[mask_indices] = per_func(obj_values)
        obj_hail_sizes = np.zeros(obj_percentiles.shape)
        model_name = self.model_name.replace(" ", "-")
        self.units = "mm"
        self.data = np.zeros((self.forecast_hours.size,
                              self.mapping_data["lon"].shape[0],
                              self.mapping_data["lon"].shape[1]), dtype=np.float32)
        sh = self.forecast_hours.min()
        for p in range(obj_hail_sizes.shape[0]):
            if self.hail_forecast_table.loc[p, self.condition_model_name.replace(" ", "-") + "_conditionthresh"] > 0.5:
                patch_mask = np.where(self.nc_patches["masks"][p] == 1)
                obj_hail_sizes[p,
                               patch_mask[0],
                               patch_mask[1]] = gamma.ppf(obj_percentiles[p,
                                                                          patch_mask[0],
                                                                          patch_mask[1]],
                                                          self.hail_forecast_table.loc[p,
                                                                                       model_name + "_shape"],
                                                          self.hail_forecast_table.loc[p,
                                                                                       model_name + "_location"],
                                                          self.hail_forecast_table.loc[p,
                                                                                       model_name + "_scale"])
                self.data[self.nc_patches["forecast_hour"][p] - sh,
                          self.nc_patches["i"][p, patch_mask[0], patch_mask[1]],
                          self.nc_patches["j"][p, patch_mask[0], patch_mask[1]]] = obj_hail_sizes[p, patch_mask[0],
                                                                                                  patch_mask[1]]
        return

    def neighborhood_probability(self, threshold, radius):
        """
        Calculate a probability based on the number of grid points in an area that exceed a threshold.

        Args:
            threshold: intensity threshold
            radius: radius of neighborhood

        Returns:

        """
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
        """
        Aggregates gridded hail sizes across time and generates neighborhood probability that maximizes threshold

        Args:
            threshold (float): intensity threshold
            radius (int): radius of influence in grid points.

        Returns:

        """
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

    def period_surrogate_severe_prob(self, threshold, radius, sigma, stagger):
        """
        Calculate surrogate severe probability for a member using method from Sobash et al. (2011).

        Args:
            threshold: intensity threshold
            radius: Radius in grid cells for neighborhood aggregation
            sigma: standard deviation of Gaussian smoother
            stagger: how many grid points to skip when reducing grid size.

        Returns:
            surrogate_grid: grid with single member storm surrogate probabilities.
        """
        i_grid, j_grid = np.indices(self.data.shape[1:])
        max_data = self.data.max(axis=0)
        max_points = np.array(np.where(max_data >= threshold)).T
        max_tree = cKDTree(max_points)
        stagger_points = np.vstack((i_grid[::stagger, ::stagger].ravel(), j_grid[::stagger, ::stagger].ravel())).T
        valid_stagger_points = np.zeros(stagger_points.shape[0])
        stagger_tree = cKDTree(stagger_points)
        hit_points = np.unique(np.concatenate(max_tree.query_ball_tree(stagger_tree, radius)))
        valid_stagger_points[hit_points] += 1
        surrogate_grid = valid_stagger_points.reshape(i_grid[::stagger, ::stagger].shape)
        surrogate_grid = gaussian_filter(surrogate_grid, sigma)
        return surrogate_grid

    def encode_grib2_percentile(self):
        """
        Encodes member percentile data to GRIB2 format.

        Returns:
            Series of GRIB2 messages
        """
        if self.data is None:
            return None
        lscale = 1e6
        grib_id_start = [7, 0, 14, 14, 2]
        gdsinfo = np.array([0, np.product(self.data.shape[-2:]), 0, 0, 30], dtype=np.int32)
        lon_0 = self.proj_dict["lon_0"]
        sw_lon = self.grid_dict["sw_lon"]
        if lon_0 < 0:
            lon_0 += 360
        if sw_lon < 0:
            sw_lon += 360
        gdtmp1 = [1, 0, self.proj_dict['a'], 0, float(self.proj_dict['a']), 0, float(self.proj_dict['b']),
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
        grib_objects = pd.Series(index=self.times, data=[None] * self.times.size, dtype=object)
        drtmp1 = np.array([0, 0, 4, 8, 0], dtype=np.int32)
        for t, time in enumerate(self.times):
            time_list = list(self.run_date.utctimetuple()[0:6])
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

    def encode_grib2_data(self):
        """
        Encodes deterministic member predictions to GRIB2 format.

        Returns:
            Series of GRIB2 messages
        """
        if self.data is None:
            return None 
        lscale = 1e6
        grib_id_start = [7, 0, 14, 14, 2]
        gdsinfo = np.array([0, np.product(self.data.shape[-2:]), 0, 0, 30], dtype=np.int32)
        lon_0 = self.proj_dict["lon_0"]
        sw_lon = self.grid_dict["sw_lon"]
        if lon_0 < 0:
            lon_0 += 360
        if sw_lon < 0:
            sw_lon += 360
        gdtmp1 = [1, 0, self.proj_dict['a'], 0, float(self.proj_dict['a']), 0, float(self.proj_dict['b']),
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
                           1                 # Number of ensemble members
                           ], dtype=np.int32)
        grib_objects = pd.Series(index=self.times, data=[None] * self.times.size, dtype=object)
        drtmp1 = np.array([0, 0, 4, 8, 0], dtype=np.int32)
        for t, time in enumerate(self.times):
            time_list = list(self.run_date.utctimetuple()[0:6])
            if grib_objects[time] is None:
                grib_objects[time] = Grib2Encode(0, np.array(grib_id_start + time_list + [2, 1], dtype=np.int32))
                grib_objects[time].addgrid(gdsinfo, gdtmp1)
            pdtmp1[8] = (time.to_pydatetime() - self.run_date).total_seconds() / 3600.0
            data = self.data[t] / 1000.0
            data[np.isnan(data)] = 0
            masked_data = np.ma.array(data, mask=data <= 0)
            pdtmp1[-2] = 0
            grib_objects[time].addfield(1, pdtmp1, 0, drtmp1, masked_data)
        return grib_objects

    def write_grib2_files(self, grib_objects, path):
        """
        Write a grib2 object to disk.

        Args:
            grib_objects: A Series of grib objects indexed by forecast time
            path: Path where grib files are written.

        """
        for t, time in enumerate(self.times.to_pydatetime()):
            grib_objects[time].end()
            filename = path + "{0}_{1}_{2}_{3}_{4}.grib2".format(self.ensemble_name,
                                                                 self.member,
                                                                 self.model_name.replace(" ", "-"),
                                                                 self.variable,
                                                                 self.run_date.strftime("%Y%m%d%H") +
                                                                 "f{0:02d}".format(self.forecast_hours[t])
                                                                 )
            fo = open(filename, "wb")
            fo.write(grib_objects[time].msg)
            fo.close()
