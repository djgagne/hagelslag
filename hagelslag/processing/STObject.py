import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from skimage.morphology import convex_hull_image
import json


class STObject(object):
    """
    The STObject stores data and location information for objects extracted from the ensemble grids.
    
    :param grid: All of the data values. Supports a 2D array of values, a list of 2D arrays, or a 3D array.
    :type grid: Numpy array or list of Numpy arrays.
    :param mask: Grid of 1's and 0's in which 1's indicate the location of the object.
    :type mask: Same as grid.
    :param x: Array of x-coordinate values in meters. Longitudes can also be placed here.
    :type x: Numpy float array.
    :param y: Array of y-coordinate values in meters. Latitudes can also be placed here.
    :type y: Numpy float array.
    :param i: Array of row indices from the full model domain.
    :type i: Numpy int array
    :param j: Array of row indices from the full model domain.
    :type j: Numpy int array
    :param start_time: The first time of the object existence.
    :type start_time: int
    :param end_time: The last time of the object existence.
    :type end_time: int
    :param step: number of hours between timesteeps
    :param dx: grid spacing
    :param u: storm motion in x-direction
    :param v: storm motion in y-direction
    """

    def __init__(self, grid, mask, x, y, i, j, start_time, end_time, step=1, dx=4000, u=None, v=None):
        if hasattr(grid, "shape") and len(grid.shape) == 2:
            self.timesteps = [grid]
            self.masks = [mask]
            self.x = [x]
            self.y = [y]
            self.i = [i]
            self.j = [j]
        elif hasattr(grid, "shape") and len(grid.shape) > 2:
            self.timesteps = []
            self.masks = []
            self.x = []
            self.y = []
            self.i = []
            self.j = []
            for l in range(grid.shape[0]):
                self.timesteps.append(grid[l])
                self.masks.append(mask[l])
                self.x.append(x[l])
                self.y.append(y[l])
                self.i.append(i[l])
                self.j.append(j[l])
        else:
            self.timesteps = grid
            self.masks = mask
            self.x = x
            self.y = y
            self.i = i
            self.j = j
        if u is not None and v is not None:
            self.u = u
            self.v = v
        else:
            self.u = np.zeros(len(self.timesteps))
            self.v = np.zeros(len(self.timesteps))
        self.dx = dx
        self.start_time = start_time
        self.end_time = end_time
        self.step = step
        self.times = np.arange(start_time, end_time + step, step)
        self.attributes = {}
        self.observations = None

    @property
    def __str__(self):
        com_x, com_y = self.center_of_mass(self.start_time)
        data = dict(maxsize=self.max_size(), comx=com_x, comy=com_y, start=self.start_time, end=self.end_time)
        return "ST Object [maxSize=%(maxsize)d,initialCenter=%(comx)0.2f,%(comy)0.2f,duration=%(start)02d-%(end)02d]" %\
               data

    def center_of_mass(self, time):
        """
        Calculate the center of mass at a given timestep.

        :param time: Time at which the center of mass calculation is performed
        :return: The x- and y-coordinates of the center of mass.
        """
        if self.start_time <= time <= self.end_time:
            diff = time - self.start_time
            valid = np.flatnonzero(self.masks[diff] != 0)
            if valid.size > 0:
                com_x = 1.0 / self.timesteps[diff].ravel()[valid].sum() * np.sum(self.timesteps[diff].ravel()[valid] *
                                                                                 self.x[diff].ravel()[valid])
                com_y = 1.0 / self.timesteps[diff].ravel()[valid].sum() * np.sum(self.timesteps[diff].ravel()[valid] *
                                                                                 self.y[diff].ravel()[valid])
            else:
                com_x = np.mean(self.x[diff])
                com_y = np.mean(self.y[diff])
        else:
            com_x = None
            com_y = None
        return com_x, com_y

    def closest_distance(self, time, other_object, other_time):
        ti = np.where(self.times == time)[0]
        oti = np.where(other_object.times == other_time)[0]
        xs = self.x[ti][self.masks[ti] == 1]
        xs = xs.reshape(xs.size, 1)
        ys = self.y[ti][self.masks[ti] == 1]
        ys = ys.reshape(ys.size, 1)
        o_xs = other_object.x[oti][other_object.masks[oti] == 1]
        o_xs = o_xs.reshape(1, o_xs.size)
        o_ys = other_object.y[oti][other_object.masks[oti] == 1]
        o_ys = o_ys.reshape(1, o_ys.size)
        distances = (xs - o_xs) ** 2 + (ys - o_ys) ** 2
        return np.sqrt(distances.min())

    def percentile_distance(self, time, other_object, other_time, percentile):
        ti = np.where(self.times == time)[0]
        oti = np.where(other_object.times == other_time)[0]
        xs = self.x[ti][self.masks[ti] == 1]
        xs = xs.reshape(xs.size, 1)
        ys = self.y[ti][self.masks[ti] == 1]
        ys = ys.reshape(ys.size, 1)
        o_xs = other_object.x[oti][other_object.masks[oti] == 1]
        o_xs = o_xs.reshape(1, o_xs.size)
        o_ys = other_object.y[oti][other_object.masks[oti] == 1]
        o_ys = o_ys.reshape(1, o_ys.size)
        distances = (xs - o_xs) ** 2 + (ys - o_ys) ** 2
        return np.sqrt(np.percentile(distances, percentile))

    def trajectory(self):
        traj = np.zeros((2, self.times.size))
        for t, time in enumerate(self.times):
            traj[:, t] = self.center_of_mass(time)
        return traj

    def get_corner(self, time):
        """
        Gets the corner array indices of the STObject at a given time that corresponds 
        to the upper left corner of the bounding box for the STObject.

        :param time: time at which the corner is being extracted.
        :return:  
        """
        if self.start_time <= time <= self.end_time:
            diff = time - self.start_time
            return self.i[diff][0, 0], self.j[diff][0, 0]
        else:
            return -1, -1

    def size(self, time):
        """
        Gets the size of the object at a given time.
        
        :param time: Time value being queried.
        :return: size of the object in pixels
        """
        if self.start_time <= time <= self.end_time:
            return self.masks[time - self.start_time].sum()
        else:
            return 0

    def max_size(self):
        """
        Gets the largest size of the object over all timesteps.
        
        :return: Maximum size of the object in pixels
        """
        sizes = np.array([m.sum() for m in self.masks])
        return sizes.max()

    def max_intensity(self, time):
        """
        Calculate the maximum intensity found at a timestep.

        """
        ti = np.where(time == self.times)[0]
        return self.timesteps[ti].max()

    def extend(self, step):
        """
        Adds the data from another STObject to this object.
        
        :param step: another STObject being added after the current one in time.
        """
        self.timesteps.extend(step.timesteps)
        self.masks.extend(step.masks)
        self.x.extend(step.x)
        self.y.extend(step.y)
        self.i.extend(step.i)
        self.j.extend(step.j)
        self.end_time = step.end_time
        self.times = np.arange(self.start_time, self.end_time + self.step, self.step)
        self.u = np.concatenate((self.u, step.u))
        self.v = np.concatenate((self.v, step.v))
        for attr in self.attributes.keys():
            if attr in step.attributes.keys():
                self.attributes[attr].extend(step.attributes[attr])

    def boundary_polygon(self, time):
        """
        Get coordinates of object boundary in counter-clockwise order
        """
        ti = np.where(time == self.times)[0]
        com_x, com_y = self.center_of_mass(time)
        boundary_image = find_boundaries(convex_hull_image(self.masks[ti]), mode='inner')
        boundary_x = self.x[ti].ravel()[boundary_image.ravel()]
        boundary_y = self.y[ti].ravel()[boundary_image.ravel()]
        r = np.sqrt((boundary_x - com_x) ** 2 + (boundary_y - com_y) ** 2)
        theta = np.arctan2((boundary_y - com_y), (boundary_x - com_x)) * 180.0 / np.pi + 360
        polar_coords = np.array([(r[x], theta[x]) for x in range(r.size)], dtype=[('r', 'f4'), ('theta', 'f4')])
        coord_order = np.argsort(polar_coords, order=['theta', 'r'])
        ordered_coords = np.vstack([boundary_x[coord_order], boundary_y[coord_order]])
        return ordered_coords

    def estimate_motion(self, time, intensity_grid, max_u, max_v):
        """
        Estimate the motion of the object with cross-correlation on the intensity values from the previous time step.

        :param time: time being evaluated.
        :param intensity_grid: 2D array of intensities used in cross correlation.
        :param max_u: Maximum x-component of motion. Used to limit search area.
        :param max_v: Maximum y-component of motion. Used to limit search area
        :return: u, v, and the minimum error.
        """
        ti = np.where(time == self.times)[0]
        i_vals = self.i[ti][self.masks[ti] == 1]
        j_vals = self.j[ti][self.masks[ti] == 1]
        obj_vals = self.timesteps[ti][self.masks[ti] == 1]
        u_shifts = np.arange(-max_u, max_u + 1)
        v_shifts = np.arange(-max_v, max_v + 1)
        min_error = 99999999999.0
        best_u = 0
        best_v = 0
        for u in u_shifts:
            j_shift = j_vals - u
            for v in v_shifts:
                i_shift = i_vals - v
                if np.all((0 <= i_shift) & (i_shift < intensity_grid.shape[0]) &
                                  (0 <= j_shift) & (j_shift < intensity_grid.shape[1])):
                    shift_vals = intensity_grid[i_shift, j_shift]
                else:
                    shift_vals = np.zeros(i_shift.shape)
                error = np.abs(shift_vals - obj_vals).mean()
                if error < min_error:
                    min_error = error
                    best_u = u * self.dx
                    best_v = v * self.dx
        if min_error > 60:
            best_u = 0
            best_v = 0
        self.u[ti] = best_u
        self.v[ti] = best_v
        return best_u, best_v, min_error

    def count_overlap(self, time, other_object, other_time):
        """
        Counts the number of points that overlap between this STObject and another STObject. Used for tracking.
        """
        ti = np.where(time == self.times)[0]
        oti = np.where(other_time == other_object.times)[0]
        obj_coords = np.zeros(self.masks[ti].sum(), dtype=[('x', int), ('y', int)])
        other_obj_coords = np.zeros(other_object.masks[oti].sum(), dtype=[('x', int), ('y', int)])
        obj_coords['x'] = self.i[ti][self.masks[ti] == 1]
        obj_coords['y'] = self.j[ti][self.masks[ti] == 1]
        other_obj_coords['x'] = other_object.i[oti][other_object.masks[oti] == 1]
        other_obj_coords['y'] = other_object.j[oti][other_object.masks[oti] == 1]
        return float(np.intersect1d(obj_coords,
                                    other_obj_coords).size) / np.maximum(self.masks[ti].sum(),
                                                                         other_object.masks[oti].sum())

    def extract_attribute_grid(self, model_grid, potential=False):
        """
        Extracts the data from an SSEFModelGrid or SSEFModelSubset within the bounding box region of the STObject.
        
        :param model_grid: A ModelGrid Object
        """

        if potential:
            var_name = model_grid.variable + "-potential"
            timesteps = np.arange(self.start_time - 1, self.end_time)
        else:
            var_name = model_grid.variable
            timesteps = np.arange(self.start_time, self.end_time + 1)
        self.attributes[var_name] = []
        for ti, t in enumerate(timesteps):
            self.attributes[var_name].append(
                model_grid.data[t - model_grid.start_hour, self.i[ti], self.j[ti]])

    def extract_tendency_grid(self, model_grid):
        var_name = model_grid.variable + "-tendency"
        self.attributes[var_name] = []
        timesteps = np.arange(self.start_time, self.end_time + 1)
        for ti, t in enumerate(timesteps):
            t_index = t - model_grid.start_hour
            self.attributes[var_name].append(
                model_grid.data[t_index, self.i[ti], self.j[ti]] - model_grid.data[t_index - 1, self.i[ti], self.j[ti]]
                )

    def calc_attribute_statistics(self, statistic_name):
        """
        Calculates summary statistics over the domains of each attribute.
        
        :param statistic_name: (string) numpy statistic
        :return: dict of statistics from each attribute grid.
        """
        stats = {}
        for var, grids in self.attributes.iteritems():
            if len(grids) > 1:
                stats[var] = getattr(np.array([getattr(np.ma.array(x, mask=self.masks[t] == 0), statistic_name)()
                                               for t, x in enumerate(grids)]), statistic_name)()
            else:
                stats[var] = getattr(np.ma.array(grids[0], mask=self.masks[0] == 0), statistic_name)()
        return stats

    def calc_attribute_statistic(self, attribute, statistic, time):
        """
        Calculate statistics based on the values of an attribute.

        :param attribute:
        :param statistic:
        :param time:
        :return:
        """
        ti = np.where(self.times == time)[0][0]
        if statistic in ['mean', 'max', 'min', 'std', 'ptp']:
            stat_val = getattr(self.attributes[attribute][ti][self.masks[ti] == 1], statistic)()
        elif statistic == 'median':
            stat_val = np.median(self.attributes[attribute][ti][self.masks[ti] == 1])
        elif statistic == "skew":
            stat_val = np.mean(self.attributes[attribute][ti][self.masks[ti] == 1]) - \
                       np.median(self.attributes[attribute][ti][self.masks[ti] == 1])
        elif 'percentile' in statistic:
            per = int(statistic.split("_")[1])
            stat_val = np.percentile(self.attributes[attribute][ti][self.masks[ti] == 1], per)
        elif 'dt' in statistic:
            stat_name = statistic[:-3]
            if ti == 0:
                stat_val = 0
            else:
                stat_val = self.calc_attribute_statistic(attribute, stat_name, time) \
                    - self.calc_attribute_statistic(attribute, stat_name, time - 1)
        else:
            stat_val = np.nan
        return stat_val

    def calc_timestep_statistic(self, statistic, time):
        """
        Calculate statistics from the primary attribute of the StObject.

        :param statistic:
        :param time:
        :return:
        """
        ti = np.where(self.times == time)[0]
        if statistic in ['mean', 'max', 'min', 'std', 'ptp']:
            stat_val = getattr(self.timesteps[ti][self.masks[ti] == 1], statistic)()
        elif statistic == 'median':
            stat_val = np.median(self.timesteps[ti][self.masks[ti] == 1])
        elif 'percentile' in statistic:
            per = int(statistic.split("_")[1])
            stat_val = np.percentile(self.timesteps[ti][self.masks[ti] == 1], per)
        elif 'dt' in statistic:
            stat_name = statistic[:-3]
            if ti == 0:
                stat_val = 0
            else:
                stat_val = self.calc_timestep_statistic(stat_name, time) -\
                    self.calc_timestep_statistic(stat_name, time - 1)
        else:
            stat_val = np.nan
        return stat_val

    def calc_shape_statistics(self, stat_names):
        """
        Calculate shape statistics using regionprops applied to the object mask.
        
        :param stat_names: List of statistics to be extracted from those calculated by regionprops.
        """
        stats = {}
        try:
            all_props = [regionprops(m) for m in self.masks]
        except TypeError:
            print(self.masks)
            exit()
        for stat in stat_names:
            stats[stat] = np.mean([p[0][stat] for p in all_props])
        return stats

    def calc_shape_step(self, stat_names, time):
        ti = np.where(self.times == time)[0]
        props = regionprops(self.masks[ti], self.timesteps[ti])[0]
        shape_stats = []
        for stat_name in stat_names:
            if "moments_hu" in stat_name:
                hu_index = int(stat_name.split("_")[-1])
                hu_name = "_".join(stat_name.split("_")[:-1])
                hu_val = np.log(props[hu_name][hu_index])
                if np.isnan(hu_val):
                    shape_stats.append(0)
                else:
                    shape_stats.append(hu_val)
            else:
                shape_stats.append(props[stat_name])
        return shape_stats

    def to_geojson(self, filename, proj, metadata=dict()):
        """
        Output the data in the STObject to a geoJSON file.

        :param filename: Name of the file
        :param proj: PyProj object for converting the x and y coordinates back to latitude and longitue values.
        :param metadata: Metadata describing the object to be included in the top-level properties.
        :return:
        """
        json_obj = {"type": "FeatureCollection", "features": [], "properties": {}}
        json_obj['properties']['times'] = self.times.tolist()
        json_obj['properties']['dx'] = self.dx
        json_obj['properties']['step'] = self.step
        json_obj['properties']['u'] = self.u.tolist()
        json_obj['properties']['v'] = self.v.tolist()
        for k, v in metadata.iteritems():
            json_obj['properties'][k] = v
        for t, time in enumerate(self.times):
            feature = {"type": "Feature",
                       "geometry": {"type": "Polygon"},
                       "properties": {}}
            boundary_coords = self.boundary_polygon(time)
            lonlat = np.vstack(proj(boundary_coords[0], boundary_coords[1], inverse=True))
            lonlat_list = lonlat.T.tolist()
            if len(lonlat_list) > 0:
                lonlat_list.append(lonlat_list[0])
            feature["geometry"]["coordinates"] = [lonlat_list]
            for attr in ["timesteps", "masks", "x", "y", "i", "j"]:
                feature["properties"][attr] = getattr(self, attr)[t].tolist()
            feature["properties"]["attributes"] = {}
            for attr_name, steps in self.attributes.iteritems():
                feature["properties"]["attributes"][attr_name] = steps[t].tolist()
            json_obj['features'].append(feature)
        file_obj = open(filename, "w")
        json.dump(json_obj, file_obj, indent=1, sort_keys=True)
        file_obj.close()
        return


def read_geojson(filename):
    """
    Reads a geojson file containing an STObject and initializes a new STObject from the information in the file.

    :param filename: Name of the geojson file
    :return: an STObject
    """
    json_file = open(filename)
    data = json.load(json_file)
    json_file.close()
    times = data["properties"]["times"]
    main_data = dict(timesteps=[], masks=[], x=[], y=[], i=[], j=[])
    attribute_data = dict()
    for feature in data["features"]:
        for main_name in main_data.iterkeys():
            main_data[main_name].append(np.array(feature["properties"][main_name]))
        for k, v in feature["properties"]["attributes"].iteritems():
            if k not in attribute_data.keys():
                attribute_data[k] = [np.array(v)]
            else:
                attribute_data[k].append(np.array(v))
    kwargs = {}
    for kw in ["dx", "step", "u", "v"]:
        if kw in data["properties"].keys():
            kwargs[kw] = data["properties"][kw]
    sto = STObject(main_data["timesteps"], main_data["masks"], main_data["x"], main_data["y"],
                   main_data["i"], main_data["j"], times[0], times[-1], **kwargs)
    for k, v in attribute_data.iteritems():
        sto.attributes[k] = v
    return sto

