import numpy as np
from STObject import STObject
from munkres import Munkres


def main():
    from datetime import datetime, timedelta
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from data.SSEFModelGrid import SSEFModelGrid
    from EnhancedWatershedSegmenter import EnhancedWatershed
    from data.MESHGrid import MESHInterpolatedGrid
    from scipy.ndimage import find_objects, gaussian_filter

    model_path = "/sharp/djgagne/spring2014/"
    member = "cn"
    start_date = datetime(2014, 6, 3, 18)
    end_date = datetime(2014, 6, 4, 6)
    date = start_date.strftime("%Y%m%d")
    variable = "cqgmax"
    start_hour = 18
    end_hour = 30
    max_motion = 30
    ew = EnhancedWatershed(5, 2, 50, 150, 20)
    ew_obs = EnhancedWatershed(5, 1, 100, 150, 50)
    min_size = 100
    om = ObjectMatcher([shifted_centroid_distance, closest_distance],
                       np.array([70.0, 30.0]),
                       np.array([16000., 1000.0]))
    model_output = SSEFModelGrid(model_path, member, date, start_hour - 1, end_hour, variable)
    print "SSEF Max:", model_output.data.max()
    obs = MESHInterpolatedGrid(start_date - timedelta(seconds=3600), end_date)
    print "MESH Info", obs.MESH.shape, obs.MESH.max(), obs.MESH.min()
    tracked_model_objects = []
    tracked_obs_objects = []
    hour_model_objects = []
    hour_obs_objects = []
    for h, hour in enumerate(np.arange(start_hour, end_hour + 1)):
        print "Finding", hour
        model_labels = ew.size_filter(ew.label(gaussian_filter(model_output.data[h + 1], 2)), min_size)
        obs_labels = ew_obs.size_filter(ew_obs.label(gaussian_filter(obs.MESH[h + 1], 2)), min_size)
        print "Num objects", model_labels.max()
        model_obj_slices = find_objects(model_labels)
        obs_obj_slices = find_objects(obs_labels)
        hour_model_objects.append([])
        hour_obs_objects.append([])
        if len(model_obj_slices) > 0:
            for sl in model_obj_slices:
                hour_model_objects[-1].append(STObject(model_output.data[h + 1][sl],
                                                       np.where(model_labels[sl] > 0, 1, 0),
                                                       model_output.x[sl],
                                                       model_output.y[sl],
                                                       model_output.i[sl],
                                                       model_output.j[sl],
                                                       hour,
                                                       hour))
                dims = hour_model_objects[-1][-1].timesteps[0].shape
                u, v, error = hour_model_objects[-1][-1].estimate_motion(hour, model_output.data[h], dims[1], dims[0])
                print "Model U: ", u, "V: ", v, "Error: ", error
        if len(obs_obj_slices) > 0:
            for sl in obs_obj_slices:
                hour_obs_objects[-1].append(STObject(obs.MESH[h + 1][sl],
                                                     np.where(obs_labels[sl] > 0, 1, 0),
                                                     model_output.x[sl],
                                                     model_output.y[sl],
                                                     model_output.i[sl],
                                                     model_output.j[sl],
                                                     hour,
                                                     hour))
                dims = hour_obs_objects[-1][-1].timesteps[0].shape
                u, v, error = hour_obs_objects[-1][-1].estimate_motion(hour, obs.MESH[h], dims[1], dims[0])
                print "Obs U: ", u, "V: ", v, error
    for h, hour in enumerate(np.arange(start_hour, end_hour + 1)):
        print "Tracking", hour
        past_time_objs = []
        for obj in tracked_model_objects:
            if obj.end_time == hour - 1:
                past_time_objs.append(obj)
        if len(past_time_objs) == 0:
            tracked_model_objects.extend(hour_model_objects[h])
        elif len(past_time_objs) > 0 and len(hour_model_objects[h]) > 0:
            assignments = om.match_objects(past_time_objs, hour_model_objects[h], hour - 1, hour)
            unpaired = range(len(hour_model_objects[h]))
            for pair in assignments:
                past_time_objs[pair[0]].extend(hour_model_objects[h][pair[1]])
                unpaired.remove(pair[1])
            if len(unpaired) > 0:
                for up in unpaired:
                    tracked_model_objects.append(hour_model_objects[h][up])
        print "Tracked Model Objects", len(tracked_model_objects)

        past_time_objs = []
        for obj in tracked_obs_objects:
            if obj.end_time == hour - 1:
                past_time_objs.append(obj)
        if len(past_time_objs) == 0:
            tracked_obs_objects.extend(hour_obs_objects[h])
        elif len(past_time_objs) > 0 and len(hour_obs_objects[h]) > 0:
            assignments = om.match_objects(past_time_objs, hour_obs_objects[h], hour - 1, hour)
            unpaired = range(len(hour_obs_objects[h]))
            for pair in assignments:
                past_time_objs[pair[0]].extend(hour_obs_objects[h][pair[1]])
                unpaired.remove(pair[1])
            if len(unpaired) > 0:
                for up in unpaired:
                    tracked_obs_objects.append(hour_obs_objects[h][up])
        print "Tracked Obs Objects", len(tracked_obs_objects)

    track_matcher = TrackMatcher([mean_minimum_centroid_distance, mean_min_time_distance],
                                 np.array([0.6, 0.4]),
                                 np.array([160000, 5]))
    track_matches = track_matcher.match_tracks(tracked_model_objects, tracked_obs_objects)

    model_trajectories = [obj.trajectory() for obj in tracked_model_objects]
    obs_trajectories = [obj.trajectory() for obj in tracked_obs_objects]

    plt.figure(figsize=(10, 6))
    model_output.basemap.drawstates()
    model_output.basemap.drawcoastlines()
    model_output.basemap.drawcountries()
    colors = ListedColormap(np.random.rand(len(model_trajectories), 3))
    for t, traj in enumerate(model_trajectories):
        for obj in tracked_model_objects:
            for t_obj in range(len(obj.times)):
                bound_coords = obj.boundary_polygon(obj.times[t_obj])
                plt.fill(bound_coords[0], bound_coords[1], fc=colors.colors[t])
                # plt.pcolormesh(obj.x[t_obj],
                #               obj.y[t_obj],
                #               np.ma.array(obj.timesteps[t_obj],mask=obj.masks[t_obj]==0),
                #               vmin=25,vmax=150,cmap=plt.get_cmap("YlOrRd",10))
        plt.plot(traj[0], traj[1], lw=2, marker='o', markersize=3, color=colors.colors[t])
        plt.plot(traj[0][0], traj[1][0], marker='^', markersize=5, color=colors.colors[t])
        plt.plot(traj[0][-1], traj[1][-1], marker='s', markersize=5, color=colors.colors[t])
    unmatched_tracks = range(len(obs_trajectories))
    for pair in track_matches:
        obs_t = pair[1]
        mod_t = pair[0]
        unmatched_tracks.remove(obs_t)
        plt.plot(obs_trajectories[obs_t][0],
                 obs_trajectories[obs_t][1],
                 lw=4,
                 marker='o',
                 markersize=3,
                 color=colors.colors[mod_t])
        plt.plot(obs_trajectories[obs_t][0][0],
                 obs_trajectories[obs_t][1][0],
                 marker='^',
                 markersize=5,
                 color=colors.colors[mod_t])
        plt.plot(obs_trajectories[obs_t][0][-1],
                 obs_trajectories[obs_t][1][-1],
                 marker='s',
                 markersize=5,
                 color=colors.colors[mod_t])
    for obs_t in unmatched_tracks:
        unmatched_tracks.remove(obs_t)
        plt.plot(obs_trajectories[obs_t][0],
                 obs_trajectories[obs_t][1],
                 lw=4,
                 marker='o',
                 markersize=3,
                 color="black")
        plt.plot(obs_trajectories[obs_t][0][0],
                 obs_trajectories[obs_t][1][0],
                 marker='^',
                 markersize=5,
                 color="black")
        plt.plot(obs_trajectories[obs_t][0][-1],
                 obs_trajectories[obs_t][1][-1],
                 marker='s',
                 markersize=5,
                 color="black")

    plt.show()
    return


class ObjectMatcher(object):
    """
    ObjectMatcher calculates distances between two sets of objects and determines the optimal object assignments
    based on the Hungarian object matching algorithm. ObjectMatcher supports the use of the weighted average of
    multiple cost functions to determine the distance between objects. Upper limits to each distance component are used
    to exclude the matching of objects that are too far apart.

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
        costs = np.zeros((len(set_a), len(set_b)))
        for a, item_a in enumerate(set_a):
            for b, item_b in enumerate(set_b):
                costs[a, b] = self.total_cost_function(item_a, item_b, time_a, time_b)
        return costs

    def total_cost_function(self, item_a, item_b, time_a, time_b):
        distances = np.zeros(len(self.weights))
        for c, component in enumerate(self.cost_function_components):
            distances[c] = component(item_a, time_a, item_b, time_b, self.max_values[c])
        total_distance = np.sum(self.weights * distances)
        return total_distance


class TrackMatcher(object):
    """
    TrackMatcher

    """

    def __init__(self, cost_function_components, weights, max_values):
        self.cost_function_components = cost_function_components
        self.weights = weights if weights.sum() == 1 else weights / weights.sum()
        self.max_values = max_values

    def match_tracks(self, set_a, set_b):
        costs = self.track_cost_matrix(set_a, set_b) * 100
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
        total_distance = np.sum(self.weights * distances)
        return total_distance


def centroid_distance(item_a, time_a, item_b, time_b, max_value):
    """
    Euclidean distance between the centroids of item_a and item_b.

    :param item_a:
    :param time_a:
    :param item_b:
    :param time_b:
    :param max_value:
    :return:
    """
    ax, ay = item_a.center_of_mass(time_a)
    bx, by = item_b.center_of_mass(time_b)
    return np.minimum(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2), max_value) / float(max_value)


def shifted_centroid_distance(item_a, time_a, item_b, time_b, max_value):
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
    times_a = item_a.times.reshape((item_a.times.size, 1))
    times_b = item_b.times.reshape((1, item_b.times.size))
    distance_matrix = (times_a - times_b) ** 2
    mean_min_distances = np.sqrt(distance_matrix.min(axis=0).mean() + distance_matrix.min(axis=1).mean())
    return mean_min_distances / float(max_value)


if __name__ == "__main__":
    main()
