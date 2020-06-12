from .STObject import STObject
from .EnhancedWatershedSegmenter import EnhancedWatershed
from .Watershed import Watershed
from .Hysteresis import Hysteresis
from hagelslag.processing.ObjectMatcher import ObjectMatcher
from scipy.ndimage import find_objects, center_of_mass, gaussian_filter
import numpy as np


def label_storm_objects(data, method, min_intensity, max_intensity, min_area=1, max_area=100, max_range=1,
                        increment=1, gaussian_sd=0):
    """
    From a 2D grid or time series of 2D grids, this method labels storm objects with either the Enhanced Watershed,
    Watershed, or Hysteresis methods.

    Args:
        data: the gridded data to be labeled. Should be a 2D numpy array in (y, x) coordinate order or a 3D numpy array
            in (time, y, x) coordinate order
        method: "ew" for Enhanced Watershed, "ws" for regular watershed, and "hyst" for hysteresis
        min_intensity: Minimum intensity threshold for gridpoints contained within any objects
        max_intensity: For watershed, any points above max_intensity are considered as the same value as max intensity.
            For hysteresis, all objects have to contain at least 1 pixel that equals or exceeds this value
        min_area: (default 1) The minimum area of any object in pixels.
        max_area: (default 100) The area threshold in pixels at which the enhanced watershed ends growth. Object area
            may exceed this threshold if the pixels at the last watershed level exceed the object area.
        max_range: Maximum difference in bins for search before growth is stopped.
        increment: Discretization increment for the enhanced watershed
        gaussian_sd: Standard deviation of Gaussian filter applied to data
    Returns:
        label_grid: an ndarray with the same shape as data in which each pixel is labeled with a positive integer value.
    """
    if method.lower() == "ew":
        labeler = EnhancedWatershed(min_intensity, increment, max_intensity, max_area, max_range)
    elif method.lower() == "ws":
        labeler = Watershed(min_intensity, max_intensity)
    else:
        labeler = Hysteresis(min_intensity, max_intensity)
    if len(data.shape) == 2:
        if gaussian_sd > 0:
            label_grid = labeler.label(gaussian_filter(data, gaussian_sd))
        else:
            label_grid = labeler.label(data)
        label_grid[data < min_intensity] = 0
        if min_area > 1:
            label_grid = labeler.size_filter(label_grid, min_area)
    else:
        label_grid = np.zeros(data.shape, dtype=np.int32)
        for t in range(data.shape[0]):
            if gaussian_sd > 0:
                label_grid[t] = labeler.label(gaussian_filter(data[t], gaussian_sd))
            else:
                label_grid[t] = labeler.label(data[t])
            label_grid[t][data[t] < min_intensity] = 0
            if min_area > 1:
                label_grid[t] = labeler.size_filter(label_grid[t], min_area)
    print("Found {0:02d}".format(int(label_grid.max())) + " storm objects.")
    return label_grid


def extract_storm_objects(label_grid, data, x_grid, y_grid, times, dx=1, dt=1, obj_buffer=0):
    """
    After storms are labeled, this method extracts the storm objects from the grid and places them into STObjects.
    The STObjects contain intensity, location, and shape information about each storm at each timestep.

    Args:
        label_grid: 2D or 3D array output by label_storm_objects.
        data: 2D or 3D array used as input to label_storm_objects.
        x_grid: 2D array of x-coordinate data, preferably on a uniform spatial grid with units of length.
        y_grid: 2D array of y-coordinate data.
        times: List or array of time values, preferably as integers
        dx: grid spacing in same units as x_grid and y_grid.
        dt: period elapsed between times
        obj_buffer: number of extra pixels beyond bounding box of object to store in each STObject

    Returns:
        storm_objects: list of lists containing STObjects identified at each time.
    """
    storm_objects = []
    if len(label_grid.shape) == 3:
        ij_grid = np.indices(label_grid.shape[1:])
        for t, time in enumerate(times):
            storm_objects.append([])
            object_slices = list(find_objects(label_grid[t], label_grid[t].max()))
            if len(object_slices) > 0:
                for o, obj_slice in enumerate(object_slices):
                    if obj_buffer > 0:
                        obj_slice_buff = [slice(np.maximum(0, osl.start - obj_buffer),
                                                np.minimum(osl.stop + obj_buffer, label_grid.shape[l + 1]))
                                          for l, osl in enumerate(obj_slice)]
                    else:
                        obj_slice_buff = obj_slice
                    storm_objects[-1].append(STObject(data[t][obj_slice_buff],
                                                      np.where(label_grid[t][obj_slice_buff] == o + 1, 1, 0),
                                                      x_grid[obj_slice_buff],
                                                      y_grid[obj_slice_buff],
                                                      ij_grid[0][obj_slice_buff],
                                                      ij_grid[1][obj_slice_buff],
                                                      time,
                                                      time,
                                                      dx=dx,
                                                      step=dt))
                    if t > 0:
                        dims = storm_objects[-1][-1].timesteps[0].shape
                        storm_objects[-1][-1].estimate_motion(time, data[t - 1], dims[1], dims[0])
    else:
        ij_grid = np.indices(label_grid.shape)
        storm_objects.append([])
        object_slices = list(find_objects(label_grid, label_grid.max()))
        if len(object_slices) > 0:
            for o, obj_slice in enumerate(object_slices):
                if obj_buffer > 0:
                    obj_slice_buff = [slice(np.maximum(0, osl.start - obj_buffer),
                                            np.minimum(osl.stop + obj_buffer, label_grid.shape[l + 1]))
                                      for l, osl in enumerate(obj_slice)]
                else:
                    obj_slice_buff = obj_slice
                storm_objects[-1].append(STObject(data[obj_slice_buff],
                                                  np.where(label_grid[obj_slice_buff] == o + 1, 1, 0),
                                                  x_grid[obj_slice_buff],
                                                  y_grid[obj_slice_buff],
                                                  ij_grid[0][obj_slice_buff],
                                                  ij_grid[1][obj_slice_buff],
                                                  times,
                                                  times,
                                                  dx=dx,
                                                  step=dt))
    return storm_objects


def extract_storm_patches(label_grid, data, x_grid, y_grid, times, dx=1, dt=1, patch_radius=16):
    """
    After storms are labeled, this method extracts boxes of equal size centered on each storm from the grid and places
    them into STObjects. The STObjects contain intensity, location, and shape information about each storm
    at each timestep.

    Args:
        label_grid: 2D or 3D array output by label_storm_objects.
        data: 2D or 3D array used as input to label_storm_objects.
        x_grid: 2D array of x-coordinate data, preferably on a uniform spatial grid with units of length.
        y_grid: 2D array of y-coordinate data.
        times: List or array of time values, preferably as integers
        dx: grid spacing in same units as x_grid and y_grid.
        dt: period elapsed between times
        patch_radius: Number of grid points from center of mass to extract

    Returns:
        storm_objects: list of lists containing STObjects identified at each time.
    """
    storm_objects = []
    if len(label_grid.shape) == 3:
        ij_grid = np.indices(label_grid.shape[1:])
        for t, time in enumerate(times):
            storm_objects.append([])
            # object_slices = find_objects(label_grid[t], label_grid[t].max())
            centers = list(center_of_mass(data[t], labels=label_grid[t], index=np.arange(1, label_grid[t].max() + 1)))
            if len(centers) > 0:
                for o, center in enumerate(centers):
                    int_center = np.round(center).astype(int)
                    obj_slice_buff = (slice(int_center[0] - patch_radius, int_center[0] + patch_radius),
                                      slice(int_center[1] - patch_radius, int_center[1] + patch_radius))
                    storm_objects[-1].append(STObject(data[t][obj_slice_buff],
                                                      np.where(label_grid[t][obj_slice_buff] == o + 1, 1, 0),
                                                      x_grid[obj_slice_buff],
                                                      y_grid[obj_slice_buff],
                                                      ij_grid[0][obj_slice_buff],
                                                      ij_grid[1][obj_slice_buff],
                                                      time,
                                                      time,
                                                      dx=dx,
                                                      step=dt))
                    if t > 0:
                        dims = storm_objects[-1][-1].timesteps[0].shape
                        storm_objects[-1][-1].estimate_motion(time, data[t - 1], dims[1], dims[0])
    else:
        ij_grid = np.indices(label_grid.shape)
        storm_objects.append([])
        centers = list(center_of_mass(data, labels=label_grid, index=np.arange(1, label_grid.max() + 1)))

        if len(centers) > 0:
            for o, center in enumerate(centers):
                int_center = np.round(center).astype(int)
                obj_slice_buff = (slice(int_center[0] - patch_radius, int_center[0] + patch_radius),
                                  slice(int_center[1] - patch_radius, int_center[1] + patch_radius))
                storm_objects[-1].append(STObject(data[obj_slice_buff],
                                                  np.where(label_grid[obj_slice_buff] == o + 1, 1, 0),
                                                  x_grid[obj_slice_buff],
                                                  y_grid[obj_slice_buff],
                                                  ij_grid[0][obj_slice_buff],
                                                  ij_grid[1][obj_slice_buff],
                                                  times[0],
                                                  times[0],
                                                  dx=dx,
                                                  step=dt))
    return storm_objects


def track_storms(storm_objects, times, distance_components, distance_maxima, distance_weights, tracked_objects=None):
    """
    Given the output of extract_storm_objects, this method tracks storms through time and merges individual
    STObjects into a set of tracks.

    Args:
        storm_objects: list of list of STObjects that have not been tracked.
        times: List of times associated with each set of STObjects
        distance_components: list of function objects that make up components of distance function
        distance_maxima: array of maximum values for each distance for normalization purposes
        distance_weights: weight given to each component of the distance function. Should add to 1.
        tracked_objects: List of STObjects that have already been tracked.
    Returns:
        tracked_objects:
    """
    obj_matcher = ObjectMatcher(distance_components, distance_weights, distance_maxima)
    if tracked_objects is None:
        tracked_objects = []
    for t, time in enumerate(times):
        past_time_objects = []
        for obj in tracked_objects:
            if obj.end_time == time - obj.step:
                past_time_objects.append(obj)
        if len(past_time_objects) == 0:
            tracked_objects.extend(storm_objects[t])
        elif len(past_time_objects) > 0 and len(storm_objects[t]) > 0:
            assignments = obj_matcher.match_objects(past_time_objects, storm_objects[t], times[t-1], times[t])
            unpaired = list(range(len(storm_objects[t])))
            for pair in assignments:
                past_time_objects[pair[0]].extend(storm_objects[t][pair[1]])
                unpaired.remove(pair[1])
            if len(unpaired) > 0:
                for up in unpaired:
                    tracked_objects.append(storm_objects[t][up])
    return tracked_objects
