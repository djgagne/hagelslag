__author__ = "David John Gagne <djgagne@ou.edu>"
import numpy as np
import pandas as pd
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


class DistributedROC(object):
    """
    Store statistics for calculating receiver operating characteristic (ROC) curves and performance diagrams and permit
    easy aggregation of ROC curves from many small datasets.

    Parameters
    ----------
    thresholds : numpy.ndarray of floats
        List of probability thresholds in increasing order.
    obs_threshold : float
        Observation value used as the split point for determining positives.
    input_str : str
        String in the format output by the __str__ method so that initialization of the object can be done
        from items in a text file.
    """
    def __init__(self, thresholds=None, obs_threshold=None, input_str=None):
        self.thresholds = thresholds
        self.obs_threshold = obs_threshold
        if self.thresholds is not None:
            self.contingency_tables = pd.DataFrame(np.zeros((thresholds.size, 4), dtype=int),
                                                   columns=["TP", "FP", "FN", "TN"])
        else:
            self.contingency_tables = pd.DataFrame(columns=["TP", "FP", "FN", "TN"])
        if input_str is not None:
            self.from_str(input_str)

    def update(self, forecasts, observations):
        """
        Update the ROC curve with a set of forecasts and observations

        :param forecasts: 1D array of forecast values
        :param observations: 1D array of observation values.
        :return:
        """
        for t, threshold in enumerate(self.thresholds):
            tp = np.count_nonzero((forecasts >= threshold)
                                  & (observations >= self.obs_threshold))
            fp = np.count_nonzero((forecasts >= threshold)
                                  & (observations < self.obs_threshold))
            fn = np.count_nonzero((forecasts < threshold)
                                  & (observations >= self.obs_threshold))
            tn = np.count_nonzero((forecasts < threshold)
                                  & (observations < self.obs_threshold))
            self.contingency_tables.ix[t] += [tp, fp, fn, tn]

    def __add__(self, other):
        """
        Add two DistributedROC objects together and combine their contingency table values.

        :param other: Another DistributedROC object.
        :return:
        """
        sum_roc = DistributedROC(self.thresholds, self.obs_threshold)
        sum_roc.contingency_tables = self.contingency_tables + other.contingency_tables
        return sum_roc

    def merge(self, other_roc):
        """
        Ingest the values of another DistributedROC object into this one and update the statistics inplace.

        :param other_roc: another DistributedROC object.
        :return:
        """
        if other_roc.thresholds.size == self.thresholds.size and np.all(other_roc.thresholds == self.thresholds):
            self.contingency_tables += other_roc.contingency_tables
        else:
            print("Input table thresholds do not match.")

    def roc_curve(self):
        """
        Generate a ROC curve from the contingency table by calculating the probability of detection (TP/(TP+FN)) and the
        probability of false detection (FP/(FP+TN)).

        :return: A pandas.DataFrame containing the POD, POFD, and the corresponding probability thresholds.
        """
        pod = self.contingency_tables["TP"].astype(float) / (self.contingency_tables["TP"] +
                                                             self.contingency_tables["FN"])
        pofd = self.contingency_tables["FP"].astype(float) / (self.contingency_tables["FP"] +
                                                              self.contingency_tables["TN"])
        return pd.DataFrame({"POD": pod, "POFD": pofd, "Thresholds": self.thresholds},
                            columns=["POD", "POFD", "Thresholds"])

    def performance_curve(self):
        """
        Calculate the Probability of Detection and False Alarm Ratio in order to output a performance diagram.

        :return: pandas.DataFrame containing POD, FAR, and probability thresholds.
        """
        pod = self.contingency_tables["TP"] / (self.contingency_tables["TP"] + self.contingency_tables["FN"])
        far = self.contingency_tables["FP"] / (self.contingency_tables["FP"] + self.contingency_tables["TP"])
        return pd.DataFrame({"POD": pod, "FAR": far, "Thresholds": self.thresholds},
                            columns=["POD", "FAR", "Thresholds"])

    def auc(self):
        """
        Calculate the Area Under the ROC Curve (AUC).

        :return:
        """
        roc_curve = self.roc_curve()
        return np.abs(np.trapz(roc_curve['POD'], x=roc_curve['POFD']))

    def __str__(self):
        """
        Output the information within the DistributedROC object to a string.

        :return:
        """
        out_str = "Obs_Threshold:{0:0.2f}".format(self.obs_threshold) + ";"
        out_str += "Thresholds:" + " ".join(["{0:0.2f}".format(t) for t in self.thresholds]) + ";"
        for col in self.contingency_tables.columns:
            out_str += col + ":" + " ".join(["{0:d}".format(t) for t in self.contingency_tables[col]]) + ";"
        out_str = out_str.rstrip(";")
        return out_str

    def __repr__(self):
        return self.__str__()

    def from_str(self, in_str):
        """
        Read the object string and parse the contingency table values from it.
        :param in_str:
        :return:
        """
        parts = in_str.split(";")
        for part in parts:
            var_name, value = part.split(":")
            if var_name == "Obs_Threshold":
                self.obs_threshold = float(value)
            elif var_name == "Thresholds":
                self.thresholds = np.array(value.split(), dtype=float)
            elif var_name in ["TP", "FP", "FN", "TN"]:
                self.contingency_tables[var_name] = np.array(value.split(), dtype=int)


class Reliability(object):
    def __init__(self, forecasts, observations, thresholds, obs_threshold):
        self.forecasts = forecasts
        self.observations = observations
        self.thresholds = thresholds
        self.obs_threshold = obs_threshold
        self.pos_relative_frequency = np.zeros(self.thresholds.shape)
        self.total_relative_frequency = np.zeros(self.thresholds.shape)
        self.calc_reliability_curve()

    def calc_reliability_curve(self):
        pos_frequency = np.zeros(self.thresholds.shape)
        total_frequency = np.zeros(self.thresholds.shape)
        for t, threshold in enumerate(self.thresholds[:-1]):
            pos_frequency[t] = np.count_nonzero((threshold <= self.forecasts) &
                                                (self.forecasts < self.thresholds[t+1]) &
                                                (self.observations > self.obs_threshold))
            total_frequency[t] = np.count_nonzero((threshold <= self.forecasts) &
                                                  (self.forecasts < self.thresholds[t+1]))
            if total_frequency[t] > 0:
                self.pos_relative_frequency[t] = pos_frequency[t] / float(total_frequency[t])
                self.total_relative_frequency[t] = total_frequency / self.forecasts.size
            else:
                self.pos_relative_frequency[t] = np.nan
        self.pos_relative_frequency[-1] = np.nan

    def brier_score(self):
        obs_truth = np.where(self.observations >= self.obs_threshold, 1, 0)
        return np.mean((self.forecasts - obs_truth) ** 2)

    def brier_score_components(self):
        obs_truth = np.where(self.observations >= self.obs_threshold, 1, 0)
        climo_freq = obs_truth.sum() / float(obs_truth.size)
        total_freq = self.total_relative_frequency * self.forecasts.size
        bins = 0.5 * (self.thresholds[0:-1] + self.thresholds[1:])
        pos_rel_freq = np.where(np.isnan(self.pos_relative_frequency), 0, self.pos_relative_frequency)
        reliability = np.mean(total_freq * (bins - pos_rel_freq) ** 2)
        resolution = np.mean(total_freq * (pos_rel_freq - climo_freq) ** 2)
        uncertainty = climo_freq * (1 - climo_freq)
        return reliability, resolution, uncertainty

    def brier_skill_score(self):
        obs_truth = np.where(self.observations >= self.obs_threshold, 1, 0)
        climo_freq = obs_truth.sum() / float(obs_truth.size)
        bs_climo = np.mean((climo_freq - obs_truth) **2)
        bs = self.brier_score()
        return 1.0 - bs / bs_climo

    def __str__(self):
        return "Brier Score: {0:0.3f}, Reliability: {1:0.3f}, Resolution: {2:0.3f}, Uncertainty: {3:0.3f}".format(
            tuple([self.brier_score()] + list(self.brier_score_components())))


class DistributedReliability(object):
    """
    A container for the statistics required to generate reliability diagrams and calculate the Brier Score.

    Parameters
    ----------
    thresholds : numpy.ndarray
        Array of probability thresholds
    obs_threshold : float
        Split value for the observations
    input_str : str
        String containing information to initialize the object from a text representation.
    """
    def __init__(self, thresholds=None, obs_threshold=None, input_str=None):
        self.thresholds = thresholds
        self.obs_threshold = obs_threshold
        if self.thresholds is None:
            self.frequencies = pd.DataFrame(columns=["Total_Freq", "Positive_Freq"])
        else:
            self.frequencies = pd.DataFrame(np.zeros((self.thresholds.size, 2), dtype=int),
                                            columns=["Total_Freq", "Positive_Freq"])
        if input_str is not None:
            self.from_str(input_str)

    def update(self, forecasts, observations):
        """
        Update the statistics with a set of forecasts and observations.

        :param forecasts:
        :param observations:
        :return:
        """
        for t, threshold in enumerate(self.thresholds[:-1]):
            self.frequencies.loc[t, "Positive_Freq"] += np.count_nonzero((threshold <= forecasts) &
                                                                         (forecasts < self.thresholds[t+1]) &
                                                                         (observations >= self.obs_threshold))
            self.frequencies.loc[t, "Total_Freq"] += np.count_nonzero((threshold <= forecasts) &
                                                                      (forecasts < self.thresholds[t+1]))

    def __add__(self, other):
        """
        Add two DistributedReliability objects together and combine their values.

        :param other: a DistributedReliability object
        :return: a DistributedReliability Object
        """
        sum_rel = DistributedReliability(self.thresholds, self.obs_threshold)
        sum_rel.frequencies = self.frequencies + other.frequencies
        return sum_rel

    def merge(self, other_rel):
        if other_rel.thresholds.size == self.thresholds.size and np.all(other_rel.thresholds == self.thresholds):
            self.frequencies += other_rel.frequencies
        else:
            print("Input table thresholds do not match.")

    def reliability_curve(self):
        """
        Calculate the reliability diagram statistics.

        :return:
        """
        total = self.frequencies["Total_Freq"].sum()
        curve = pd.DataFrame(columns=["Bin_Start", "Bin_End", "Bin_Center",
                                      "Positive_Relative_Freq", "Total_Relative_Freq"])
        curve["Bin_Start"] = self.thresholds[:-1]
        curve["Bin_End"] = self.thresholds[1:]
        curve["Bin_Center"] = 0.5 * (self.thresholds[:-1] + self.thresholds[1:])
        curve["Positive_Relative_Freq"] = self.frequencies["Positive_Freq"] / self.frequencies["Total_Freq"]
        curve["Total_Relative_Freq"] = self.frequencies["Total_Freq"] / total
        return curve

    def brier_score_components(self):
        """
        Calculate the components of the Brier score decomposition: reliability, resolution, and uncertainty.

        :return:
        """
        rel_curve = self.reliability_curve()
        total = self.frequencies["Total_Freq"].sum()
        climo_freq = float(self.frequencies["Positive_Freq"].sum()) / self.frequencies["Total_Freq"].sum()
        reliability = np.sum(self.frequencies["Total_Freq"] * (rel_curve["Bin_Start"] -
                                                                rel_curve["Positive_Relative_Freq"]) ** 2) / total
        resolution = np.sum(self.frequencies["Total_Freq"] * (rel_curve["Positive_Relative_Freq"] - climo_freq) ** 2) \
                     / total
        uncertainty = climo_freq * (1 - climo_freq)
        return reliability, resolution, uncertainty

    def brier_score(self):
        """
        Calculate the Brier Score

        :return:
        """
        reliability, resolution, uncertainty = self.brier_score_components()
        return reliability - resolution + uncertainty

    def brier_skill_score(self):
        """
        Calculate the Brier Skill Score

        :return:
        """
        reliability, resolution, uncertainty = self.brier_score_components()
        return (resolution - reliability) / uncertainty

    def __str__(self):
        out_str = "Obs_Threshold:{0:0.2f}".format(self.obs_threshold) + ";"
        out_str += "Thresholds:" + " ".join(["{0:0.2f}".format(t) for t in self.thresholds]) + ";"
        for col in self.frequencies.columns:
            out_str += col + ":" + " ".join(["{0:d}".format(t) for t in self.frequencies[col]]) + ";"
        out_str = out_str.rstrip(";")
        return out_str

    def __repr__(self):
        return self.__str__()

    def from_str(self, in_str):
        parts = in_str.split(";")
        for part in parts:
            var_name, value = part.split(":")
            if var_name == "Obs_Threshold":
                self.obs_threshold = float(value)
            elif var_name == "Thresholds":
                self.thresholds = np.array(value.split(), dtype=float)
            elif var_name in ["Positive_Freq", "Total_Freq"]:
                self.frequencies[var_name] = np.array(value.split(), dtype=int)


class DistributedCRPS(object):
    """
    A container for the statistics used to calculate the Continuous Ranked Probability Score

    Parameters
    ----------
    thresholds : numpy.ndarray
        Array of the intensity threshold bins
    input_str : str
        String containing the information for initializing the object from a storable text format.
    """
    def __init__(self, thresholds=None, input_str=None):
        self.thresholds = thresholds
        if self.thresholds is None:
            self.errors = pd.DataFrame(columns=["Errors", "Pos_Counts"])
        else:
            self.errors = pd.DataFrame({"Errors":np.zeros(self.thresholds.size), 
                                        "Pos_Counts":np.zeros(self.thresholds.size, dtype=int)}, 
                                       columns=["Errors", "Pos_Counts"])
        self.num_forecasts = 0
        if input_str is not None:
            self.from_str(input_str)

    def update(self, forecasts, observations):
        """
        Update the statistics with forecasts and observations.

        :param forecasts:
        :param observations:
        :return:
        """
        forecast_cdfs = np.cumsum(forecasts, axis=1)
        obs_cdfs = np.zeros((observations.size, self.thresholds.size))
        for o, observation in enumerate(observations):
            obs_cdfs[o, self.thresholds >= observation] = 1
        self.errors["Errors"] += np.sum((forecast_cdfs - obs_cdfs) ** 2, axis=0)
        self.errors["Pos_Counts"] += np.sum(obs_cdfs, axis=0)
        self.num_forecasts += forecasts.shape[0]

    def __add__(self, other):
        sum_crps = DistributedCRPS(self.thresholds)
        sum_crps.errors = self.errors + other.errors
        sum_crps.num_forecasts = self.num_forecasts + other.num_forecasts
        return sum_crps

    def merge(self, other_crps):
        if other_crps.thresholds.size == self.thresholds.size and np.all(other_crps.thresholds == self.thresholds):
            self.errors += other_crps.errors
            self.num_forecasts += other_crps.num_forecasts
        else:
            print("Input table thresholds do not match.")

    def crps(self):
        """
        Calculates the continuous ranked probability score.

        :return:
        """
        return self.errors["Errors"].sum() / (self.thresholds.size * self.num_forecasts)

    def from_str(self, in_str):
        str_parts = in_str.split(";")
        for part in str_parts:
            var_name, value = part.split(":")
            if var_name == "Thresholds":
                self.thresholds = np.array(value.split(), dtype=float)
            elif var_name == "Errors":
                self.errors["Errors"] = np.array(value.split(), dtype=float)
            elif var_name == "Pos_Counts":
                self.errors["Pos_Counts"] = np.array(value.split(), dtype=int)
            elif var_name == "Num_Forecasts":
                self.num_forecasts = int(value)

    def __str__(self):
        out_str = ""
        out_str += "Thresholds:" + " ".join(["{0:0.2f}".format(t) for t in self.thresholds]) + ";"
        out_str += "Errors:" + " ".join(["{0:0.3f}".format(e) for e in self.errors["Errors"]]) + ";"
        out_str += "Pos_Counts:" + " ".join("{0:d}".format(int(e)) for e in self.errors["Pos_Counts"]) + ";"
        out_str += "Num_Forecasts:{0:d}".format(self.num_forecasts)
        return out_str

    def __repr__(self):
        return self.__str__()
