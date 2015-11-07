import numpy as np
from STObject import STObject
from hagelslag.util.munkres import Munkres


class ObjectMatcher(object):
    """
    ObjectMatcher calculates distances between two sets of objects and determines the optimal object assignments
    based on the Hungarian object matching algorithm. ObjectMatcher supports the use of the weighted average of
    multiple cost functions to determine the distance between objects. Upper limits to each distance component are used
    to exclude the matching of objects that are too far apart.

    Parameters
    ----------
    cost_function_components : list
        List of distance functions for matching
    weights : list
        List of weights for each distance function
    max_values : list
        List of the maximum allowable distance for each distance function component.
    """

    def __init__(self, cost_function_components, weights, max_values):
        self.cost_function_components = cost_function_components
        self.weights = weights
        self.max_values = max_values
        if self.weights.sum() != 1:
            self.weights /= float(self.weights.sum())
        return

    def match_objects(self, set_a, set_b, time_a, time_b):
        """
        Match two sets of objects at particular times.

        :param set_a: list of STObjects
        :param set_b: list of STObjects
        :param time_a: time at which set_a is being evaluated for matching
        :param time_b: time at which set_b is being evaluated for matching
        :return: list of tuples containing (set_a index, set_b index) for each match
        """
        costs = self.cost_matrix(set_a, set_b, time_a, time_b) * 100
        min_row_costs = costs.min(axis=1)
        min_col_costs = costs.min(axis=0)
        good_rows = np.where(min_row_costs < 100)[0]
        good_cols = np.where(min_col_costs < 100)[0]
        assignments = []
        if len(good_rows) > 0 and len(good_cols) > 0:
            munk = Munkres()
            initial_assignments = munk.compute(costs[np.meshgrid(good_rows, good_cols, indexing='ij')].tolist())
            initial_assignments = [(good_rows[x[0]], good_cols[x[1]]) for x in initial_assignments]
            for a in initial_assignments:
                if costs[a[0], a[1]] < 100:
                    assignments.append(a)
        return assignments

    def cost_matrix(self, set_a, set_b, time_a, time_b):
        """
        Calculates the costs (distances) between the items in set a and set b at the specified times.

        :param set_a: List of STObjects
        :param set_b: List of STObjects
        :param time_a: time at which objects in set_a are evaluated
        :param time_b: time at whcih object in set_b are evaluated
        """
        costs = np.zeros((len(set_a), len(set_b)))
        for a, item_a in enumerate(set_a):
            for b, item_b in enumerate(set_b):
                costs[a, b] = self.total_cost_function(item_a, item_b, time_a, time_b)
        return costs

    def total_cost_function(self, item_a, item_b, time_a, time_b):
        """
        Calculate total cost function between two items.
        """
        distances = np.zeros(len(self.weights))
        for c, component in enumerate(self.cost_function_components):
            distances[c] = component(item_a, time_a, item_b, time_b, self.max_values[c])
        total_distance = np.sum(self.weights * distances)
        return total_distance


class TrackMatcher(object):
    """
    Find the optimal pairings among two sets of STObject tracks.

    """

    def __init__(self, cost_function_components, weights, max_values):
        self.cost_function_components = cost_function_components
        self.weights = weights if weights.sum() == 1 else weights / weights.sum()
        self.max_values = max_values

    def match_tracks(self, set_a, set_b, unique_matches=True):
        costs = self.track_cost_matrix(set_a, set_b) * 100
        min_row_costs = costs.min(axis=1)
        min_col_costs = costs.min(axis=0)
        good_rows = np.where(min_row_costs < 100)[0]
        good_cols = np.where(min_col_costs < 100)[0]
        assignments = []
        if len(good_rows) > 0 and len(good_cols) > 0:
            if unique_matches:
                munk = Munkres()
                initial_assignments = munk.compute(costs[np.meshgrid(good_rows, good_cols, indexing='ij')].tolist())
                initial_assignments = [(good_rows[x[0]], good_cols[x[1]]) for x in initial_assignments]
            else:
                b_matches = costs[np.meshgrid(good_rows, good_cols, indexing='ij')].argmin(axis=1)
                a_matches = np.arange(b_matches.size)
                initial_assignments = [(good_rows[a_matches[x]], good_cols[b_matches[x]])
                                       for x in range(b_matches.size)]
            for a in initial_assignments:
                if costs[a[0], a[1]] < 100:
                    assignments.append(a)
        return assignments

    def neighbor_matches(self, set_a, set_b):
        costs = self.track_cost_matrix(set_a, set_b)
        all_neighbors = []
        for i in range(len(set_a)):
            neighbors = np.where(costs[i] < 1)[0].tolist()
            if len(neighbors) > 0:
                all_neighbors.append((i, tuple(neighbors)))
        return all_neighbors

    def track_cost_matrix(self, set_a, set_b):
        costs = np.zeros((len(set_a), len(set_b)))
        for a, item_a in enumerate(set_a):
            for b, item_b in enumerate(set_b):
                costs[a, b] = self.track_cost_function(item_a, item_b)
        return costs

    def track_cost_function(self, item_a, item_b):
        distances = np.zeros(len(self.weights))
        for c, component in enumerate(self.cost_function_components):
            distances[c] = component(item_a, item_b, self.max_values[c])
        if np.all(distances < 1):
            total_distance = np.sum(self.weights * distances)
        else:
            total_distance = 1.0
        return total_distance


def centroid_distance(item_a, time_a, item_b, time_b, max_value):
    """
    Euclidean distance between the centroids of item_a and item_b.

    :param item_a: STObject
    :param time_a: STObject
    :param item_b:
    :param time_b:
    :param max_value:
    :return:
    """
    ax, ay = item_a.center_of_mass(time_a)
    bx, by = item_b.center_of_mass(time_b)
    return np.minimum(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2), max_value) / float(max_value)


def shifted_centroid_distance(item_a, time_a, item_b, time_b, max_value):
    """
    Centroid distance with motion corrections.

    :param item_a:
    :param time_a:
    :param item_b:
    :param time_b:
    :param max_value:
    :return:
    """
    ax, ay = item_a.center_of_mass(time_a)
    bx, by = item_b.center_of_mass(time_b)
    if time_a < time_b:
        bx = bx - item_b.u
        by = by - item_b.v
    else:
        ax = ax - item_a.u
        ay = ay - item_a.v
    return np.minimum(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2), max_value) / float(max_value)


def closest_distance(item_a, time_a, item_b, time_b, max_value):
    """
    Euclidean distance between the pixels in item_a and item_b closest to each other.
    """
    return np.minimum(item_a.closest_distance(time_a, item_b, time_b), max_value) / float(max_value)


def percentile_distance(item_a, time_a, item_b, time_b, max_value, percentile=2):
    return np.minimum(item_a.percentile_distance(time_a, item_b, time_b, percentile), max_value) / float(max_value)


def ellipse_distance(item_a, time_a, item_b, time_b, max_value):
    """
    Calculate differences in the properties of ellipses fitted to each object.
    """
    ts = np.array([0, np.pi])
    ell_a = item_a.get_ellipse_model(time_a)
    ell_b = item_b.get_ellipse_model(time_b)
    ends_a = ell_a.predict_xy(ts)
    ends_b = ell_b.predict_xy(ts)
    distances = np.sqrt((ends_a[:, 0:1] - ends_b[:, 0:1].T) ** 2 + (ends_a[:, 1:] - ends_b[:, 1:].T) ** 2)
    return np.minimum(distances[0, 1], max_value) / float(max_value)


def nonoverlap(item_a, time_a, item_b, time_b, max_value):
    """
    Percentage of pixels in each object that do not overlap with the other object
    """
    return np.minimum(1 - item_a.count_overlap(time_a, item_b, time_b), max_value)


def max_intensity(item_a, time_a, item_b, time_b, max_value):
    """
    RMS Difference in intensities.
    """
    intensity_a = item_a.max_intensity(time_a)
    intensity_b = item_b.max_intensity(time_b)
    diff = np.sqrt((intensity_a - intensity_b) ** 2)
    return np.minimum(diff, max_value) / float(max_value)


def area_difference(item_a, time_a, item_b, time_b, max_value):
    """
    RMS Difference in object areas.
    """
    size_a = item_a.size(time_a)
    size_b = item_b.size(time_b)
    diff = np.sqrt((size_a - size_b) ** 2)
    return np.minimum(diff, max_value) / float(max_value)


def mean_minimum_centroid_distance(item_a, item_b, max_value):
    """
    RMS difference in the minimum distances from the centroids of one track to the centroids of another track
    """
    centroids_a = np.array([item_a.center_of_mass(t) for t in item_a.times])
    centroids_b = np.array([item_b.center_of_mass(t) for t in item_b.times])
    distance_matrix = (centroids_a[:, 0:1] - centroids_b.T[0:1]) ** 2 + (centroids_a[:, 1:] - centroids_b.T[1:]) ** 2
    mean_min_distances = np.sqrt(distance_matrix.min(axis=0).mean() + distance_matrix.min(axis=1).mean())
    return mean_min_distances / float(max_value)


def mean_min_time_distance(item_a, item_b, max_value):
    """
    Calculate the mean time difference among the time steps in each object.

    :param item_a: STObject
    :param item_b: STObject
    :param max_value: maximum value of the distance
    :return:
    """
    times_a = item_a.times.reshape((item_a.times.size, 1))
    times_b = item_b.times.reshape((1, item_b.times.size))
    distance_matrix = (times_a - times_b) ** 2
    mean_min_distances = np.sqrt(distance_matrix.min(axis=0).mean() + distance_matrix.min(axis=1).mean())
    return mean_min_distances / float(max_value)


def duration_distance(item_a, item_b, max_value):
    """
    Absolute difference in the duration of two items

    :param item_a: STObject
    :param item_b: STObject
    :param max_value: maximum value of the distance
    :return:
    """
    duration_a = item_a.times.size
    duration_b = item_b.times.size
    return np.abs(duration_a - duration_b) / float(max_value)
