import numpy as np
from hagelslag.util.munkres import Munkres
import pandas as pd


class ObjectMatcher(object):
    """
    ObjectMatcher calculates distances between two sets of objects and determines the optimal object assignments
    based on the Hungarian object matching algorithm. ObjectMatcher supports the use of the weighted average of
    multiple cost functions to determine the distance between objects. Upper limits to each distance component are used
    to exclude the matching of objects that are too far apart.

    Attributes:
        cost_function_components: List of distance functions for matching
        weights: List of weights for each distance function
        max_values : List of the maximum allowable distance for each distance function component.
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

        Args:
            set_a: list of STObjects
            set_b: list of STObjects
            time_a: time at which set_a is being evaluated for matching
            time_b: time at which set_b is being evaluated for matching

        Returns:
            List of tuples containing (set_a index, set_b index) for each match
        """
        costs = self.cost_matrix(set_a, set_b, time_a, time_b) * 100
        min_row_costs = costs.min(axis=1)
        min_col_costs = costs.min(axis=0)
        good_rows = np.where(min_row_costs < 100)[0]
        good_cols = np.where(min_col_costs < 100)[0]
        assignments = []
        if len(good_rows) > 0 and len(good_cols) > 0:
            munk = Munkres()
            initial_assignments = munk.compute(costs[tuple(np.meshgrid(good_rows, good_cols, indexing='ij'))].tolist())
            initial_assignments = [(good_rows[x[0]], good_cols[x[1]]) for x in initial_assignments]
            for a in initial_assignments:
                if costs[a[0], a[1]] < 100:
                    assignments.append(a)
        return assignments

    def cost_matrix(self, set_a, set_b, time_a, time_b):
        """
        Calculates the costs (distances) between the items in set a and set b at the specified times.

        Args:
            set_a: List of STObjects
            set_b: List of STObjects
            time_a: time at which objects in set_a are evaluated
            time_b: time at whcih object in set_b are evaluated

        Returns:
            A numpy array with shape [len(set_a), len(set_b)] containing the cost matrix between the items in set a
            and the items in set b.
        """
        costs = np.zeros((len(set_a), len(set_b)))
        for a, item_a in enumerate(set_a):
            for b, item_b in enumerate(set_b):
                costs[a, b] = self.total_cost_function(item_a, item_b, time_a, time_b)
        return costs

    def total_cost_function(self, item_a, item_b, time_a, time_b):
        """
        Calculate total cost function between two items.

        Args:
            item_a: STObject
            item_b: STObject
            time_a: Timestep in item_a at which cost function is evaluated
            time_b: Timestep in item_b at which cost function is evaluated

        Returns:
            The total weighted distance between item_a and item_b
        """
        distances = np.zeros(len(self.weights))
        for c, component in enumerate(self.cost_function_components):
            distances[c] = component(item_a, time_a, item_b, time_b, self.max_values[c])
        total_distance = np.sum(self.weights * distances)
        return total_distance


class TrackMatcher(object):
    """
    Find the optimal pairings among two sets of STObject tracks.

    Attributes:
        cost_function_components: Array of cost function objects
        weights: Array of weights for each cost function. All should sum to 1.
        max_values: Array of distance values that correspond to the upper limit distance that should be
            considered.

    """

    def __init__(self, cost_function_components, weights, max_values):
        self.cost_function_components = cost_function_components
        self.weights = weights if weights.sum() == 1 else weights / weights.sum()
        self.max_values = max_values

    def match_tracks(self, set_a, set_b, closest_matches=False):
        """
        Find the optimal set of matching assignments between set a and set b. This function supports optimal 1:1
        matching using the Munkres method and matching from every object in set a to the closest object in set b.
        In this situation set b accepts multiple matches from set a.

        Args:
            set_a:
            set_b:
            closest_matches:

        Returns:

        """
        costs = self.track_cost_matrix(set_a, set_b) * 100
        min_row_costs = costs.min(axis=1)
        min_col_costs = costs.min(axis=0)
        good_rows = np.where(min_row_costs < 100)[0]
        good_cols = np.where(min_col_costs < 100)[0]
        assignments = []
        if len(good_rows) > 0 and len(good_cols) > 0:
            if closest_matches:
                b_matches = costs[np.meshgrid(good_rows, good_cols, indexing='ij')].argmin(axis=1)
                a_matches = np.arange(b_matches.size)
                initial_assignments = [(good_rows[a_matches[x]], good_cols[b_matches[x]])
                                       for x in range(b_matches.size)]
            else:
                munk = Munkres()
                initial_assignments = munk.compute(costs[np.meshgrid(good_rows, good_cols, indexing='ij')].tolist())
                initial_assignments = [(good_rows[x[0]], good_cols[x[1]]) for x in initial_assignments]
            for a in initial_assignments:
                if costs[a[0], a[1]] < 100:
                    assignments.append(a)
        return assignments

    def raw_cost_matrix(self, set_a, set_b):
        cost_matrix = np.zeros((len(set_a), len(set_b), len(self.cost_function_components)))
        for (a, b, c), x in np.ndenumerate(cost_matrix):
            cost_matrix[a, b, c] = self.cost_function_components[c](set_a[a],
                                                                    set_b[b], self.max_values[c]) * self.max_values[c]
        return cost_matrix

    def neighbor_matches(self, set_a, set_b):
        costs = self.track_cost_matrix(set_a, set_b)
        all_neighbors = []
        for i in range(len(set_a)):
            neighbors = np.where(costs[i] < 1)[0]
            sorted_neighbors = neighbors[costs[i][neighbors].argsort()]
            if len(neighbors) > 0:
                all_neighbors.append((i, tuple(sorted_neighbors)))
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


class TrackStepMatcher(object):
    """
    Determine if each step in a track is in close proximity to steps from another set of tracks
    """
    def __init__(self, cost_function_components, max_values):
        self.cost_function_components = cost_function_components
        self.max_values = max_values

    def match(self, set_a, set_b):
        """
        For each step in each track from set_a, identify all steps in all tracks from set_b that meet all
        cost function criteria
        
        Args:
            set_a: List of STObjects
            set_b: List of STObjects

        Returns:
            track_pairings: pandas.DataFrame 
        """
        track_step_matches = [[] * len(set_a)]

        costs = self.cost_matrix(set_a, set_b)
        valid_costs = np.all(costs < 1, axis=2)
        set_a_matches, set_b_matches = np.where(valid_costs)
        s = 0
        track_pairings = pd.DataFrame(index=np.arange(costs.shape[0]),
                                      columns=["Track", "Step", "Time", "Matched", "Pairings"], dtype=object)
        set_b_info = []
        for trb, track_b in enumerate(set_b):
            for t, time in enumerate(track_b.times):
                set_b_info.append((trb, t))
        set_b_info_arr = np.array(set_b_info, dtype=int)
        for tr, track_a in enumerate(set_a):
            for t, time in enumerate(track_a.times):
                track_pairings.loc[s, ["Track", "Step", "Time"]] = [tr, t, time]
                track_pairings.loc[s, "Matched"] = 1 if np.count_nonzero(set_a_matches == s) > 0 else 0
                if track_pairings.loc[s, "Matched"] == 1:
                    track_pairings.loc[s, "Pairings"] = set_b_info_arr[set_b_matches[set_a_matches == s]]
                else:
                    track_pairings.loc[s, "Pairings"] = np.array([])
                s += 1
        return track_pairings

    def cost_matrix(self, set_a, set_b):
        num_steps_a = np.sum([track_a.times.size for track_a in set_a])
        num_steps_b = np.sum([track_b.times.size for track_b in set_b])
        cost_matrix = np.zeros((num_steps_a, num_steps_b, len(self.cost_function_components)))
        a_i = 0
        for a, track_a in enumerate(set_a):
            for time_a in track_a.times:
                b_i = 0
                for b, track_b in enumerate(set_b):
                    for time_b in track_b.times:
                        cost_matrix[a_i, b_i] = self.cost(track_a, time_a, track_b, time_b)
                        b_i += 1
                a_i += 1
        return cost_matrix

    def cost(self, track_a, time_a, track_b, time_b):
        return np.array([cost_func(track_a, time_a, track_b, time_b, self.max_values[c])
                         for c, cost_func in enumerate(self.cost_function_components)])


def centroid_distance(item_a, time_a, item_b, time_b, max_value):
    """
    Euclidean distance between the centroids of item_a and item_b.

    Args:
        item_a: STObject from the first set in ObjectMatcher
        time_a: Time integer being evaluated
        item_b: STObject from the second set in ObjectMatcher
        time_b: Time integer being evaluated
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    ax, ay = item_a.center_of_mass(time_a)
    bx, by = item_b.center_of_mass(time_b)
    return np.minimum(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2), max_value) / float(max_value)


def time_distance(item_a, time_a, item_b, time_b, max_value):
    return np.minimum(np.abs(time_b - time_a), max_value) / float(max_value)


def shifted_centroid_distance(item_a, time_a, item_b, time_b, max_value):
    """
    Centroid distance with motion corrections.

    Args:
        item_a: STObject from the first set in ObjectMatcher
        time_a: Time integer being evaluated
        item_b: STObject from the second set in ObjectMatcher
        time_b: Time integer being evaluated
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
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

    Args:
        item_a: STObject from the first set in ObjectMatcher
        time_a: Time integer being evaluated
        item_b: STObject from the second set in ObjectMatcher
        time_b: Time integer being evaluated
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    return np.minimum(item_a.closest_distance(time_a, item_b, time_b), max_value) / float(max_value)


def ellipse_distance(item_a, time_a, item_b, time_b, max_value):
    """
    Calculate differences in the properties of ellipses fitted to each object.

    Args:
        item_a: STObject from the first set in ObjectMatcher
        time_a: Time integer being evaluated
        item_b: STObject from the second set in ObjectMatcher
        time_b: Time integer being evaluated
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
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

    Args:
        item_a: STObject from the first set in ObjectMatcher
        time_a: Time integer being evaluated
        item_b: STObject from the second set in ObjectMatcher
        time_b: Time integer being evaluated
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    return np.minimum(1 - item_a.count_overlap(time_a, item_b, time_b), max_value) / float(max_value)


def max_intensity(item_a, time_a, item_b, time_b, max_value):
    """
    RMS difference in maximum intensity

    Args:
        item_a: STObject from the first set in ObjectMatcher
        time_a: Time integer being evaluated
        item_b: STObject from the second set in ObjectMatcher
        time_b: Time integer being evaluated
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    intensity_a = item_a.max_intensity(time_a)
    intensity_b = item_b.max_intensity(time_b)
    diff = np.sqrt((intensity_a - intensity_b) ** 2)
    return np.minimum(diff, max_value) / float(max_value)


def area_difference(item_a, time_a, item_b, time_b, max_value):
    """
    RMS Difference in object areas.

    Args:
        item_a: STObject from the first set in ObjectMatcher
        time_a: Time integer being evaluated
        item_b: STObject from the second set in ObjectMatcher
        time_b: Time integer being evaluated
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    size_a = item_a.size(time_a)
    size_b = item_b.size(time_b)
    diff = np.sqrt((size_a - size_b) ** 2)
    return np.minimum(diff, max_value) / float(max_value)


def mean_minimum_centroid_distance(item_a, item_b, max_value):
    """
    RMS difference in the minimum distances from the centroids of one track to the centroids of another track

    Args:
        item_a: STObject from the first set in TrackMatcher
        item_b: STObject from the second set in TrackMatcher
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    centroids_a = np.array([item_a.center_of_mass(t) for t in item_a.times])
    centroids_b = np.array([item_b.center_of_mass(t) for t in item_b.times])
    distance_matrix = (centroids_a[:, 0:1] - centroids_b.T[0:1]) ** 2 + (centroids_a[:, 1:] - centroids_b.T[1:]) ** 2
    mean_min_distances = np.sqrt(distance_matrix.min(axis=0).mean() + distance_matrix.min(axis=1).mean())
    return np.minimum(mean_min_distances, max_value) / float(max_value)


def mean_min_time_distance(item_a, item_b, max_value):
    """
    Calculate the mean time difference among the time steps in each object.

    Args:
        item_a: STObject from the first set in TrackMatcher
        item_b: STObject from the second set in TrackMatcher
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    times_a = item_a.times.reshape((item_a.times.size, 1))
    times_b = item_b.times.reshape((1, item_b.times.size))
    distance_matrix = (times_a - times_b) ** 2
    mean_min_distances = np.sqrt(distance_matrix.min(axis=0).mean() + distance_matrix.min(axis=1).mean())
    return np.minimum(mean_min_distances, max_value) / float(max_value)


def start_centroid_distance(item_a, item_b, max_value):
    """
    Distance between the centroids of the first step in each object.

    Args:
        item_a: STObject from the first set in TrackMatcher
        item_b: STObject from the second set in TrackMatcher
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    start_a = item_a.center_of_mass(item_a.times[0])
    start_b = item_b.center_of_mass(item_b.times[0])
    start_distance = np.sqrt((start_a[0] - start_b[0]) ** 2 + (start_a[1] - start_b[1]) ** 2)
    return np.minimum(start_distance, max_value) / float(max_value)


def start_time_distance(item_a, item_b, max_value):
    """
    Absolute difference between the starting times of each item.

    Args:
        item_a: STObject from the first set in TrackMatcher
        item_b: STObject from the second set in TrackMatcher
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    start_time_diff = np.abs(item_a.times[0] - item_b.times[0])
    return np.minimum(start_time_diff, max_value) / float(max_value)


def duration_distance(item_a, item_b, max_value):
    """
    Absolute difference in the duration of two items

    Args:
        item_a: STObject from the first set in TrackMatcher
        item_b: STObject from the second set in TrackMatcher
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    duration_a = item_a.times.size
    duration_b = item_b.times.size
    return np.minimum(np.abs(duration_a - duration_b), max_value) / float(max_value)


def mean_area_distance(item_a, item_b, max_value):
    """
    Absolute difference in the means of the areas of each track over time.

    Args:
        item_a: STObject from the first set in TrackMatcher
        item_b: STObject from the second set in TrackMatcher
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    mean_area_a = np.mean([item_a.size(t) for t in item_a.times])
    mean_area_b = np.mean([item_b.size(t) for t in item_b.times])
    return np.abs(mean_area_a - mean_area_b) / float(max_value)
