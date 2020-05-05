import numpy as np
import pandas as pd
from pyproj import Proj
import pygrib
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from os.path import exists


class HailForecastGrid(object):
    """
    HailForecastGrid loads and stores gridded machine learning hail forecasts from GRIB2 files. It can load
    an arbitrary number of members and timesteps at once.

    Attributes:
        run_date (datetime.datetime): Date of the initial time of the model run
        start_date (datetime.datetime): Date of the initial forecast time being loaded
        end_date (datetime.datetime): Date of the final forecast time being loaded
        forecast_dates (pandas.DatetimeIndex): All forecast times
        ensemble_name (str): Name of the NWP ensemble being used
        ml_model (str): Name of the machine learning model being loaded
        variable (str): Name of the machine learning model variable being forecast
        message_number (int): Field in the GRIB2 file to load. The first field in the file has message number 1.
        path (str): Path to top-level GRIB2 directory. Assumes files are stored in directories by run_date
        data (ndarray): Hail forecast data with dimensions (member, time, y, x)
        lon (ndarray): 2D array of longitudes
        lat (ndarray): 2D array of latitudes
        x (ndarray): 2d array of x-coordinate values in km
        y (ndarray): 2d array of y-coordinate values in km
        i (ndarray): 2d array of row indices
        j (ndarray): 2d array of column indices
        dx (float): distance between grid points
        proj (Proj): a pyproj projection object used for converting lat-lon points to x-y coordinate values
        projparams (dict): PROJ4 parameters describing map projection
    """
    def __init__(self, run_date, start_date, end_date, ensemble_name, ml_model, members,
                 variable, message_number, path):
        self.run_date = run_date
        self.start_date = start_date
        self.end_date = end_date
        self.forecast_dates = pd.date_range(start=self.start_date, end=self.end_date, freq="1H")
        self.ensemble_name = ensemble_name
        self.ml_model = ml_model
        self.members = members
        self.variable = variable
        self.message_number = message_number
        self.path = path
        self.data = None
        self.lon = None
        self.lat = None
        self.x = None
        self.y = None
        self.i = None
        self.j = None
        self.dx = None
        self.proj = None
        self.projparams = None
        return

    def load_data(self):
        for m, member in enumerate(self.members):
            for f, forecast_date in enumerate(self.forecast_dates.to_pydatetime()):
                dt = int((forecast_date - self.run_date).total_seconds() / 3600)
                filename_args = (self.ensemble_name, member, self.ml_model, self.variable,
                                 forecast_date.strftime("%Y%m%d%H%M"))
                filename = self.path + self.run_date.strftime("%Y%m%d") + \
                           "/{0}_{1}_{2}_{3}_{4}.grib2".format(*filename_args)
                if not exists(filename):
                    filename_args = (self.ensemble_name, member, self.ml_model, self.variable,
                                     self.run_date.strftime("%Y%m%d%H") + "f{0:02d}".format(dt))
                    filename = self.path + self.run_date.strftime("%Y%m%d") + \
                               "/{0}_{1}_{2}_{3}_{4}.grib2".format(*filename_args)
                    if not exists(filename):
                        continue
                grbs = pygrib.open(filename)
                if self.lon is None:
                    self.lat, self.lon = grbs[self.message_number].latlons()
                    self.projparams = grbs[self.message_number].projparams
                    self.proj = Proj(grbs[self.message_number].projparams)
                    self.x, self.y = self.proj(self.lon, self.lat)
                    self.x /= 1000.0
                    self.y /= 1000.0
                    self.dx = grbs[self.message_number]['DxInMetres'] / 1000.0
                    self.i, self.j = np.indices(self.lon.shape)
                data = grbs[self.message_number].values
                data *= 1000.0
                if self.data is None:
                    self.data = np.empty((len(self.members), len(self.forecast_dates),
                                          data.shape[0], data.shape[1]), dtype=float)
                self.data[m, f] = data.filled(0)
                grbs.close()
        return

    def period_neighborhood_probability(self, radius, smoothing, threshold, stride,start_time,end_time):
        """
        Calculate the neighborhood probability over the full period of the forecast

        Args:
            radius: circular radius from each point in km
            smoothing: width of Gaussian smoother in km
            threshold: intensity of exceedance
            stride: number of grid points to skip for reduced neighborhood grid

        Returns:
            (neighborhood probabilities)
        """
        neighbor_x = self.x[::stride, ::stride]
        neighbor_y = self.y[::stride, ::stride]
        neighbor_kd_tree = cKDTree(np.vstack((neighbor_x.ravel(), neighbor_y.ravel())).T)
        neighbor_prob = np.zeros((self.data.shape[0], neighbor_x.shape[0], neighbor_x.shape[1]))
        print('Forecast Hours: {0}-{1}'.format(start_time, end_time))
        for m in range(len(self.members)):
            period_max = self.data[m,start_time:end_time,:,:].max(axis=0)
            valid_i, valid_j = np.where(period_max >= threshold)
            print(self.members[m], len(valid_i))
            if len(valid_i) > 0:
                var_kd_tree = cKDTree(np.vstack((self.x[valid_i, valid_j], self.y[valid_i, valid_j])).T)
                exceed_points = np.unique(np.concatenate(var_kd_tree.query_ball_tree(neighbor_kd_tree, radius))).astype(int)
                exceed_i, exceed_j = np.unravel_index(exceed_points, neighbor_x.shape)
                neighbor_prob[m][exceed_i, exceed_j] = 1
                if smoothing > 0:
                    neighbor_prob[m] = gaussian_filter(neighbor_prob[m], smoothing,mode='constant')
        return neighbor_prob
