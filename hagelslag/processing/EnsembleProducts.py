from hagelslag.data.ModelOutput import ModelOutput
import numpy as np
from scipy.ndimage import convolve, maximum_filter, gaussian_filter
from skimage.morphology import disk
from netCDF4 import Dataset
import os


class EnsembleProducts(object):
    def __init__(self, ensemble_name, members, run_date, variable, start_date, end_date, path, single_step):
        self.ensemble_name = ensemble_name
        self.members = members
        self.run_date = run_date
        self.variable = variable
        self.start_date = start_date
        self.end_date = end_date
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
                                 self.run_date, self.variable + ">={0:0.2f}_{1}".format(threshold, self.units),
                                 self.start_date, self.end_date, "")

    def neighborhood_probability(self, threshold, radius, sigma=0):
        weights = disk(radius)
        neighborhood_prob = np.zeros(self.data.shape[1:])
        for t in range(self.data.shape[1]):
            for m in range(self.data.shape[0]):
                maximized = maximum_filter(np.where(self.data[m, t] >= threshold, 1, 0), weights)
                neighborhood_prob[t] += convolve(maximized, weights / float(weights.sum()), mode="constant")
            neighborhood_prob[t] /= self.data.shape[0]
            if sigma > 0:
                neighborhood_prob[t] = gaussian_filter(neighborhood_prob[t], sigma=sigma)
        return EnsembleConsensus(neighborhood_prob, "neighborhood_probability", self.ensemble_name,
                                 self.run_date, self.variable + ">={0:0.2f}_{1}".format(threshold, self.units),
                                 self.start_date, self.end_date, "")

    def period_max_neighborhood_probability(self, threshold, radius, sigma=0):
        weights = disk(radius)
        neighborhood_prob = np.zeros(self.data.shape[2:])
        for m in range(self.data.shape[0]):
            maximized = maximum_filter(np.where(self.data[m] >= threshold, 1, 0), weights).max(axis=0)
            neighborhood_prob += convolve(maximized, weights / float(weights.sum()), mode="constant")
        neighborhood_prob /= self.data.shape[0]
        if sigma > 0:
            neighborhood_prob = gaussian_filter(neighborhood_prob, sigma=sigma)
        return EnsembleConsensus(neighborhood_prob,
                                 "neighborhood_probability_{0:02d}-hour_max".format(self.data.shape[1]),
                                 self.ensemble_name,
                                 self.run_date, self.variable + ">={0.02f}_{1}".format(threshold, self.units),
                                 self.start_date, self.end_date, "")


class EnsembleConsensus(object):
    def __init__(self, data, consensus_type, ensemble_name, run_date, variable, start_date, end_date, units):
        self.data = data
        self.consensus_type = consensus_type
        self.ensemble_name = ensemble_name
        self.run_date = run_date
        self.variable = variable
        self.start_date = start_date
        self.end_date = end_date
        self.units = units

    def to_netcdf(self, filename):
        if os.access(filename, os.R_OK):
            out_data = Dataset(filename, "r+")
        else:
            out_data = Dataset(filename, "w")
        out_data.close()
        return

