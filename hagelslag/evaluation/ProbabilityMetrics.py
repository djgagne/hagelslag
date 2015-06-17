__author__ = "David John Gagne <djgagne@ou.edu>"
import numpy as np
from ContingencyTable import ContingencyTable


class ROC(object):
    def __init__(self, forecasts, observations, thresholds, obs_threshold):
        self.forecasts = forecasts
        self.observations = observations
        self.thresholds = thresholds
        self.obs_threshold = obs_threshold
        self.pod = np.zeros(thresholds.shape)
        self.pofd = np.zeros(thresholds.shape)
        self.far = np.zeros(thresholds.shape)
        self.calc_roc()

    def calc_roc(self):
        ct = ContingencyTable(0, 0, 0, 0)
        for t, threshold in enumerate(self.thresholds):
            tp = np.count_nonzero((self.forecasts >= threshold)
                                  & (self.observations >= self.obs_threshold))
            fp = np.count_nonzero((self.forecasts >= threshold)
                                  & (self.observations < self.obs_threshold))
            fn = np.count_nonzero((self.forecasts < threshold)
                                  & (self.observations >= self.obs_threshold))
            tn = np.count_nonzero((self.forecasts < threshold)
                                  & (self.observations < self.obs_threshold))
            ct.update(tp, fp, fn, tn)
            self.pod[t] = ct.pod()
            self.pofd[t] = ct.pofd()
            self.far[t] = ct.far()

    def auc(self):
        return -np.trapz(self.pod, self.pofd)


class Reliability(object):
    def __init__(self, forecasts, observations, thresholds, obs_threshold):
        self.forecasts = forecasts
        self.observations = observations
        self.thresholds = thresholds
        self.obs_threshold = obs_threshold
        self.pos_relative_frequency = np.zeros(self.thresholds.shape)
        self.total_relative_frequency = np.zeros(self.thresholds.shape)

    def calc_reliability_curve(self):
        pos_frequency = np.zeros(self.thresholds.shape)
        total_frequency = np.zeros(self.thresholds.shape)
        for t, threshold in enumerate(self.thresholds[:-1]):
            pos_frequency[t] = np.count_nonzero((threshold <= self.forecasts) &
                                                (self.forecasts < self.thresholds[t+1]) &
                                                (self.forecasts > self.obs_threshold))
            total_frequency[t] = np.count_nonzero((threshold <= self.forecasts) &
                                                  (self.forecasts < self.thresholds[t+1]))
            if total_frequency[t] > 0:
                self.pos_relative_frequency[t] = pos_frequency[t] / float(total_frequency[t])
                self.total_relative_frequency[t] = total_frequency / self.forecasts.size
            else:
                self.pos_relative_frequency[t] = np.nan
        self.pos_relative_frequency[-1] = np.nan

