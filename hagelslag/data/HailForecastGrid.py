import numpy as np
import pandas as pd
from pyproj import Proj
import pygrib
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter, gaussian_filter
from skimage.morphology import disk
from os.path import exists


class HailForecastGrid(object):
    def __init__(self, run_date, start_date, end_date, ensemble_name, ml_model, members,
                 variable, message_number, path):
        self.run_date = run_date
        self.start_date = start_date
        self.end_date = end_date
        self.forecast_dates = pd.DatetimeIndex(start=self.start_date, end=self.end_date, freq="1H")
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
                grbs = pygrib.open(filename)
                if self.lon is None:
                    self.lat, self.lon = grbs[self.message_number].latlons()
                    self.projparams = grbs[self.message_number].projparams
                    self.proj = Proj(grbs[self.message_number].projparams)
                    self.x, self.y = self.proj(self.lon, self.lat)
                    self.x /= 1000.0
                    self.y /= 1000.0
                    self.dx = grbs[self.message_number]['DxInMetres'] / 1000.0
                data = grbs[self.message_number].values
                data *= 1000.0
                data.set_fill_value(0)
                if self.data is None:
                    self.data = np.ma.empty((len(self.members), len(self.forecast_dates),
                                             data.shape[0], data.shape[1]), dtype=float)
                    self.data.set_fill_value(0)
                self.data[m, f] = data
                grbs.close()
        return

    def period_neighborhood_probability(self, radius, smoothing, threshold, stride):
        """
        Calculate the neighborhood probability over the full period of the forecast

        Args:
            radius: circular radius from each point in km
            smoothing: width of Gaussian smoother in km
            threshold: intensity of exceedance
            stride: number of grid points to skip for reduced neighborhood grid

        Returns:
            (neighborhood probablities, longitudes, latitudes)
        """
        neighbor_total = disk(int(radius / self.dx)).sum()
        neighbor_lons = self.lon[::stride, ::stride]
        neighbor_lats = self.lat[::stride, ::stride]
        neighbor_x = self.x[::stride, ::stride]
        neighbor_y = self.y[::stride, ::stride]
        neighbor_kd_tree = cKDTree(np.vstack((neighbor_x.ravel(), neighbor_y.ravel())).T)
        neighbor_prob = np.zeros(neighbor_lons.shape)
        for m in range(len(self.members)):
            period_max = self.data[m].max(axis=0, fill_value=0)
            period_max[period_max.mask == True] = 0
            period_max = maximum_filter(period_max, footprint=disk(int(radius / self.dx)), mode='constant')
            valid_i, valid_j = np.ma.where(period_max >= threshold)
            print(m, len(valid_i))
            if len(valid_i) > 0:
                var_kd_tree = cKDTree(np.vstack((self.x[valid_i, valid_j], self.y[valid_i, valid_j])).T)
                nearest_counts = np.array([len(c) for c in
                                          neighbor_kd_tree.query_ball_tree(var_kd_tree, radius, p=2, eps=0)])
                neighbor_prob += nearest_counts.reshape(neighbor_prob.shape)
        print("Max counts", neighbor_prob.max())
        neighbor_prob /= float(neighbor_total * len(self.members))
        print("Max counts divided", neighbor_prob.max(), neighbor_total, len(self.members))
        neighbor_prob = gaussian_filter(neighbor_prob, int(smoothing / self.dx / stride))
        print("Max counts smoothed", neighbor_prob.max(), smoothing, self.dx, stride, int(smoothing/self.dx/stride))
        return neighbor_prob, neighbor_lons, neighbor_lats
