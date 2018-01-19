import numpy as np
import pandas as pd
from .ContingencyTable import ContingencyTable

__author__ = "David John Gagne <djgagne@ou.edu>"


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
            tp = np.count_nonzero((self.forecasts >= threshold) & (self.observations >= self.obs_threshold))
            fp = np.count_nonzero((self.forecasts >= threshold) & (self.observations < self.obs_threshold))
            fn = np.count_nonzero((self.forecasts < threshold) & (self.observations >= self.obs_threshold))
            tn = np.count_nonzero((self.forecasts < threshold) & (self.observations < self.obs_threshold))
            ct.update(tp, fp, fn, tn)
            self.pod[t] = ct.pod()
            self.pofd[t] = ct.pofd()
            self.far[t] = ct.far()

    def auc(self):
        return -np.trapz(self.pod, self.pofd)


class DistributedROC(object):
    """
    ROC sparse representation that can be aggregated and can generate ROC curves and performance diagrams.

    A DistributedROC object is given a specified set of thresholds (could be probability or real-valued) and then
    stores a pandas DataFrame of contingency tables for each threshold. The contingency tables are updated with a
    set of forecasts and observations, but the original forecast and observation values are not kept. DistributedROC
    objects can be combined by adding them together or by storing them in an iterable and summing the contents of the
    iterable together. This is especially useful when verifying large numbers of cases in parallel.

    Attributes:
        thresholds (numpy.ndarray): List of probability thresholds in increasing order.
        obs_threshold (float):  Observation values >= obs_threshold are positive events.
        contingency_tables (pandas.DataFrame): Stores contingency table counts for each probability threshold

    Examples:

        >>> import numpy as np
        >>> forecasts = np.random.random(size=1000)
        >>> obs = np.random.random_integers(0, 1, size=1000)
        >>> roc = DistributedROC(thresholds=np.arange(0, 1.1, 0.1), obs_threshold=1)
        >>> roc.update(forecasts, obs)
        >>> print(roc.auc())
    """
    def __init__(self, thresholds=np.arange(0, 1.1, 0.1), obs_threshold=1.0, input_str=None):
        """
        Initializes the DistributedROC object. If input_str is not None, then the DistributedROC object is
         initialized with the contents of input_str. Otherwise an empty contingency table is created.

        Args:
            thresholds (numpy.array): Array of thresholds in increasing order.
            obs_threshold (float): Split threshold (>= is positive event) (< is negative event)
            input_str (None or str): String containing information for DistributedROC
        """
        self.thresholds = thresholds
        self.obs_threshold = obs_threshold
        self.contingency_tables = pd.DataFrame(np.zeros((thresholds.size, 4), dtype=int),
                                               columns=["TP", "FP", "FN", "TN"])
        if input_str is not None:
            self.from_str(input_str)

    def update(self, forecasts, observations):
        """
        Update the ROC curve with a set of forecasts and observations

        Args:
            forecasts: 1D array of forecast values
            observations: 1D array of observation values.
        """
        for t, threshold in enumerate(self.thresholds):
            tp = np.count_nonzero((forecasts >= threshold) & (observations >= self.obs_threshold))
            fp = np.count_nonzero((forecasts >= threshold) &
                                  (observations < self.obs_threshold))
            fn = np.count_nonzero((forecasts < threshold) &
                                  (observations >= self.obs_threshold))
            tn = np.count_nonzero((forecasts < threshold) &
                                  (observations < self.obs_threshold))
            self.contingency_tables.iloc[t] += [tp, fp, fn, tn]
    
    def clear(self):
        self.contingency_tables.loc[:, :] = 0

    def __add__(self, other):
        """
        Add two DistributedROC objects together and combine their contingency table values.

        Args:
            other: Another DistributedROC object.
        """
        sum_roc = DistributedROC(self.thresholds, self.obs_threshold)
        sum_roc.contingency_tables = self.contingency_tables + other.contingency_tables
        return sum_roc

    def merge(self, other_roc):
        """
        Ingest the values of another DistributedROC object into this one and update the statistics inplace.

        Args:
            other_roc: another DistributedROC object.
        """
        if other_roc.thresholds.size == self.thresholds.size and np.all(other_roc.thresholds == self.thresholds):
            self.contingency_tables += other_roc.contingency_tables
        else:
            print("Input table thresholds do not match.")

    def roc_curve(self):
        """
        Generate a ROC curve from the contingency table by calculating the probability of detection (TP/(TP+FN)) and the
        probability of false detection (FP/(FP+TN)).

        Returns:
            A pandas.DataFrame containing the POD, POFD, and the corresponding probability thresholds.
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

        Returns:
            pandas.DataFrame containing POD, FAR, and probability thresholds.
        """
        pod = self.contingency_tables["TP"] / (self.contingency_tables["TP"] + self.contingency_tables["FN"])
        far = self.contingency_tables["FP"] / (self.contingency_tables["FP"] + self.contingency_tables["TP"])
        far[(self.contingency_tables["FP"] + self.contingency_tables["TP"]) == 0] = np.nan
        return pd.DataFrame({"POD": pod, "FAR": far, "Thresholds": self.thresholds},
                            columns=["POD", "FAR", "Thresholds"])

    def auc(self):
        """
        Calculate the Area Under the ROC Curve (AUC).
        """
        roc_curve = self.roc_curve()
        return np.abs(np.trapz(roc_curve['POD'], x=roc_curve['POFD']))

    def max_csi(self):
        """
        Calculate the maximum Critical Success Index across all probability thresholds

        Returns:
            The maximum CSI as a float
        """
        csi = self.contingency_tables["TP"] / (self.contingency_tables["TP"] + self.contingency_tables["FN"] +
                                               self.contingency_tables["FP"])
        return csi.max()

    def max_threshold_score(self, score="ets"):
        cts = self.get_contingency_tables()
        scores = np.array([getattr(ct, score)() for ct in cts])
        return self.thresholds[scores.argmax()], scores.max()

    def get_contingency_tables(self):
        """
        Create an Array of ContingencyTable objects for each probability threshold.

        Returns:
            Array of ContingencyTable objects
        """
        return np.array([ContingencyTable(*ct) for ct in self.contingency_tables.values])

    def __str__(self):
        """
        Output the information within the DistributedROC object to a string.
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
        Read the DistributedROC string and parse the contingency table values from it.

        Args:
            in_str (str): The string output from the __str__ method
        """
        parts = in_str.split(";")
        for part in parts:
            var_name, value = part.split(":")
            if var_name == "Obs_Threshold":
                self.obs_threshold = float(value)
            elif var_name == "Thresholds":
                self.thresholds = np.array(value.split(), dtype=float)
                self.contingency_tables = pd.DataFrame(columns=self.contingency_tables.columns,
                                                       data=np.zeros((self.thresholds.size,
                                                                     self.contingency_tables.columns.size)))
            elif var_name in self.contingency_tables.columns:
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
        bs_climo = np.mean((climo_freq - obs_truth) ** 2)
        bs = self.brier_score()
        return 1.0 - bs / bs_climo

    def __str__(self):
        return "Brier Score: {0:0.3f}, Reliability: {1:0.3f}, Resolution: {2:0.3f}, Uncertainty: {3:0.3f}".format(
            tuple([self.brier_score()] + list(self.brier_score_components())))


class DistributedReliability(object):
    """
    A container for the statistics required to generate reliability diagrams and calculate the Brier Score.

    DistributedReliabilty objects accept binary probabilistic forecasts and associated observations. The
    forecasts are then discretized into the different probability bins. The total frequency and the frequency
    of positive events for each probability bin are tracked. The Brier Score, Brier Skill Score, and
    Brier score components can all be derived from this information. Like the DistributedROC object,
    DistributedReliability objects can be summed together, and their contents can be output as a string.

    Attributes:
        thresholds (numpy.ndarray): Array of probability thresholds
        obs_threshold (float): Split value (>=) for determining positive observation events
        frequencies (pandas.DataFrame): Stores the total and positive frequencies for each bin

    Examples:

        >>> forecasts = np.random.random(1000)
        >>> obs = np.random.random_integers(0, 1, 1000)
        >>> rel = DistributedReliability()
        >>> rel.update(forecasts, obs)
        >>> print(rel.brier_score())
    """

    def __init__(self, thresholds=np.arange(0, 1.1, 0.05), obs_threshold=1.0, input_str=None):
        """
        Initialize the DistributedReliability object.

        Args:
            thresholds (numpy.ndarray): Array of probability thresholds
            obs_threshold (float): Split value for observations
            input_str (str): String containing information to initialize the object from a text representation.
        """
        self.thresholds = thresholds
        self.obs_threshold = obs_threshold
        self.frequencies = pd.DataFrame(np.zeros((self.thresholds.size, 2), dtype=int),
                                        columns=["Total_Freq", "Positive_Freq"])
        if input_str is not None:
            self.from_str(input_str)

    def update(self, forecasts, observations):
        """
        Update the statistics with a set of forecasts and observations.

        Args:
            forecasts (numpy.ndarray): Array of forecast probability values
            observations (numpy.ndarray): Array of observation values
        """
        for t, threshold in enumerate(self.thresholds[:-1]):
            self.frequencies.loc[t, "Positive_Freq"] += np.count_nonzero((threshold <= forecasts) &
                                                                         (forecasts < self.thresholds[t+1]) &
                                                                         (observations >= self.obs_threshold))
            self.frequencies.loc[t, "Total_Freq"] += np.count_nonzero((threshold <= forecasts) &
                                                                      (forecasts < self.thresholds[t+1]))
    
    def clear(self):
        self.frequencies.loc[:, :] = 0

    def __add__(self, other):
        """
        Add two DistributedReliability objects together and combine their values.

        Args:
            other: a DistributedReliability object

        Returns:
            A DistributedReliability Object
        """
        sum_rel = DistributedReliability(self.thresholds, self.obs_threshold)
        sum_rel.frequencies = self.frequencies + other.frequencies
        return sum_rel

    def merge(self, other_rel):
        """
        Ingest another DistributedReliability and add its contents to the current object.

        Args:
            other_rel: a Distributed reliability object.
        """
        if other_rel.thresholds.size == self.thresholds.size and np.all(other_rel.thresholds == self.thresholds):
            self.frequencies += other_rel.frequencies
        else:
            print("Input table thresholds do not match.")

    def reliability_curve(self):
        """
        Calculates the reliability diagram statistics. The key columns are Bin_Start and Positive_Relative_Freq

        Returns:
            pandas.DataFrame
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

    def climatology(self):
        """
        Calculates the sample climatological relative frequency of the event being forecast.

        """
        return float(self.frequencies["Positive_Freq"].sum()) / self.frequencies["Total_Freq"].sum()

    def brier_score(self):
        """
        Calculate the Brier Score
        """
        reliability, resolution, uncertainty = self.brier_score_components()
        return reliability - resolution + uncertainty

    def brier_skill_score(self):
        """
        Calculate the Brier Skill Score
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
        """
        Updates the object attributes with the information contained in the input string

        Args:
            in_str (str): String output by the __str__ method containing all of the attribute values

        """
        parts = in_str.split(";")
        for part in parts:
            var_name, value = part.split(":")
            if var_name == "Obs_Threshold":
                self.obs_threshold = float(value)
            elif var_name == "Thresholds":
                self.thresholds = np.array(value.split(), dtype=float)
                self.frequencies = pd.DataFrame(columns=self.frequencies.columns,
                                                data=np.zeros((self.thresholds.size,
                                                              self.frequencies.columns.size)))
            elif var_name in ["Positive_Freq", "Total_Freq"]:
                self.frequencies[var_name] = np.array(value.split(), dtype=int)


class DistributedCRPS(object):
    """
    A container for the data used to calculate the Continuous Ranked Probability Score.

    Attributes:
        thresholds (numpy.ndarray): Array of the intensity threshold bins
        input_str (str): String containing the information for initializing the object
    """

    def __init__(self, thresholds=np.arange(0, 200.0), input_str=None):
        self.thresholds = thresholds
        crps_columns = ["F_2", "F_O", "O_2", "O"]
        self.errors = pd.DataFrame(columns=crps_columns,
                                   data=np.zeros((len(thresholds), len(crps_columns))), dtype=float)
        self.num_forecasts = 0
        if input_str is not None:
            self.from_str(input_str)

    def update(self, forecasts, observations):
        """
        Update the statistics with forecasts and observations.

        Args:
            forecasts: The discrete Cumulative Distribution Functions of
            observations:
        """
        if len(observations.shape) == 1:
            obs_cdfs = np.zeros((observations.size, self.thresholds.size))
            for o, observation in enumerate(observations):
                obs_cdfs[o, self.thresholds >= observation] = 1
        else:
            obs_cdfs = observations
        self.errors["F_2"] += np.sum(forecasts ** 2, axis=0)
        self.errors["F_O"] += np.sum(forecasts * obs_cdfs, axis=0)
        self.errors["O_2"] += np.sum(obs_cdfs ** 2, axis=0)
        self.errors["O"] += np.sum(obs_cdfs, axis=0)
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
            print("ERROR: Input table thresholds do not match.")

    def crps(self):
        """
        Calculates the continuous ranked probability score.
        """
        return np.sum(self.errors["F_2"].values - self.errors["F_O"].values * 2.0 + self.errors["O_2"].values) / \
            (self.thresholds.size * self.num_forecasts)

    def crps_climo(self):
        """
        Calculate the climatological CRPS.
        """
        o_bar = self.errors["O"].values / float(self.num_forecasts)
        crps_c = np.sum(self.num_forecasts * (o_bar ** 2) - o_bar * self.errors["O"].values * 2.0 +
                        self.errors["O_2"].values) / float(self.thresholds.size * self.num_forecasts)
        return crps_c

    def crpss(self):
        """
        Calculate the continous ranked probability skill score from existing data.
        """
        crps_f = self.crps()
        crps_c = self.crps_climo()
        return 1.0 - float(crps_f) / float(crps_c)

    def from_str(self, in_str):
        str_parts = in_str.split(";")
        for part in str_parts:
            var_name, value = part.split(":")
            if var_name == "Thresholds":
                self.thresholds = np.array(value.split(), dtype=float)
                self.errors = pd.DataFrame(data=np.zeros((self.thresholds.size, self.errors.columns.size)),
                                           columns=self.errors.columns)
            elif var_name in self.errors.columns:
                self.errors[var_name] = np.array(value.split(), dtype=float)
            elif var_name == "Num_Forecasts":
                self.num_forecasts = int(value)

    def __str__(self):
        out_str = ""
        out_str += "Thresholds:" + " ".join(["{0:0.2f}".format(t) for t in self.thresholds]) + ";"
        for col in self.errors.columns:
            out_str += col + ":" + " ".join(["{0:0.3f}".format(e) for e in self.errors[col]]) + ";"
        out_str += "Num_Forecasts:{0:d}".format(self.num_forecasts)
        return out_str

    def __repr__(self):
        return self.__str__()


def bootstrap(score_objs, n_boot=1000):
    """
    Given a set of DistributedROC or DistributedReliability objects, this function performs a
    bootstrap resampling of the objects and returns n_boot aggregations of them.

    Args:
        score_objs: A list of DistributedROC or DistributedReliability objects. Objects must have an __add__ method
        n_boot (int): Number of bootstrap samples

    Returns:
        An array of DistributedROC or DistributedReliability
    """
    all_samples = np.random.choice(score_objs, size=(n_boot, len(score_objs)), replace=True)
    return all_samples.sum(axis=1)
