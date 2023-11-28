import json

import numpy as np
from skimage.measure import regionprops, find_contours
from skimage.morphology import convex_hull_image
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes


class STObject(object):
    """
    The STObject stores data and location information for objects extracted from the ensemble grids.

    Attributes:
        grid (ndarray): All of the data values. Supports a 2D array of values, a list of 2D arrays, or a 3D array.
        mask (ndarray): Grid of 1's and 0's in which 1's indicate the location of the object.
        x (ndarray): Array of x-coordinate values in meters. Longitudes can also be placed here.
        y (ndarray): Array of y-coordinate values in meters. Latitudes can also be placed here.
        i (ndarray): Array of row indices from the full model domain.
        j (ndarray): Array of column indices from the full model domain.
        start_time: The first time of the object existence.
        end_time: The last time of the object existence.
        step: number of hours between timesteps
        dx: grid spacing
        u: storm motion in x-direction
        v: storm motion in y-direction
    """

    def __init__(self, grid, mask, x, y, i, j, start_time, end_time, step=1, dx=3000, u=None, v=None):
        if hasattr(grid, "shape") and len(grid.shape) == 2:
            self.timesteps = [grid]
            self.masks = [np.array(mask, dtype=int)]
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
                self.masks.append(np.array(mask[l], dtype=int))
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
        return "ST Object [maxSize=%(maxsize)d,initialCenter=%(comx)0.2f,%(comy)0.2f,duration=%(start)02d-%(end)02d]" % \
               data

    def center_of_mass(self, time):
        """
        Calculate the center of mass at a given timestep.

        Args:
            time: Time at which the center of mass calculation is performed

        Returns:
            The x- and y-coordinates of the center of mass.
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

    def center_of_mass_ij(self, time):
        """
        Calculate the center of mass in terms of row and column coordinates at a given timestep.

        Args:
            time: Time at which the center of mass calculation is performed

        Returns:
            The x- and y-coordinates of the center of mass.
        """
        if self.start_time <= time <= self.end_time:
            diff = time - self.start_time
            valid = np.flatnonzero(self.masks[diff] > 0)
            if valid.size > 0:
                com_i = 1.0 / self.timesteps[diff].ravel()[valid].sum() * np.sum(self.timesteps[diff].ravel()[valid] *
                                                                                 self.i[diff].ravel()[valid])
                com_j = 1.0 / self.timesteps[diff].ravel()[valid].sum() * np.sum(self.timesteps[diff].ravel()[valid] *
                                                                                 self.j[diff].ravel()[valid])
            else:
                com_i = np.mean(self.i[diff])
                com_j = np.mean(self.j[diff])
            com_i = int(np.round(com_i))
            com_j = int(np.round(com_j))
        else:
            com_i = None
            com_j = None
        return com_i, com_j

    def extract_patch(self, patch_radius, full_x, full_y, full_i, full_j):
        """
        Extract patch of uniform radius from existing STObject. This is intended for extracting patches from
        STObjects that are built around the bounding box of the original object. Areas outside the object are
        padded with zeros.

        Args:
            patch_radius (int): radius of patch in pixels
            full_x: full x grid encompassing the original field from which the objects have been extracted.
            full_y: full y grid
            full_i: full i (row) grid
            full_j: full j (row) grid

        Returns:
            new STObject containing the a patched slice of the original STObject.
        """
        com_ijs = [self.center_of_mass_ij(time) for time in self.times]
        patch_grid = []
        patch_mask = []
        patch_x = []
        patch_y = []
        patch_i = []
        patch_j = []
        for t, time in enumerate(self.times):
            obj_slice_buff = (slice(com_ijs[t][0] - patch_radius, com_ijs[t][0] + patch_radius),
                              slice(com_ijs[t][1] - patch_radius, com_ijs[t][1] + patch_radius))
            obj_slice_local = [[com_ijs[t][0] - self.i[t].min() - patch_radius,
                                com_ijs[t][0] - self.i[t].min() + patch_radius],
                               [com_ijs[t][1] - self.j[t].min() - patch_radius,
                                com_ijs[t][1] - self.j[t].min() + patch_radius]]
            patch_i.append(full_i[obj_slice_buff])
            patch_j.append(full_j[obj_slice_buff])
            patch_x.append(full_x[obj_slice_buff])
            patch_y.append(full_y[obj_slice_buff])
            pad_i_l = abs(obj_slice_local[0][0]) if obj_slice_local[0][0] < 0 else 0
            pad_i_u = obj_slice_local[0][1] - self.timesteps[t].shape[0] \
                if obj_slice_local[0][1] - self.timesteps[t].shape[0] > 0 else 0
            pad_j_l = abs(obj_slice_local[1][0]) if obj_slice_local[1][0] < 0 else 0
            pad_j_u = obj_slice_local[1][1] - self.timesteps[t].shape[1] \
                if obj_slice_local[1][1] - self.timesteps[t].shape[1] > 0 else 0

            if obj_slice_local[0][0] < 0:
                obj_slice_local[0][0] = 0
                obj_slice_local[0][1] += pad_i_l
            if obj_slice_local[1][0] < 0:
                obj_slice_local[1][0] = 0
                obj_slice_local[1][1] += pad_j_l
            pad_grid = np.pad(self.timesteps[t], pad_width=[(pad_i_l, pad_i_l + pad_i_u), (pad_j_l, pad_j_l + pad_j_u)])
            pad_mask = np.pad(self.masks[t], pad_width=[(pad_i_l, pad_i_l + pad_i_u), (pad_j_l, pad_j_l + pad_j_u)])
            obj_slice_const = (slice(obj_slice_local[0][0], obj_slice_local[0][1]),
                               slice(obj_slice_local[1][0], obj_slice_local[1][1]))
            patch_grid.append(pad_grid[obj_slice_const])
            patch_mask.append(pad_mask[obj_slice_const])
        patch_obj = STObject(patch_grid, patch_mask, patch_x, patch_y, patch_i, patch_j, self.start_time,
                             self.end_time, step=self.step, dx=self.dx, u=self.u, v=self.v)
        return patch_obj


    def closest_distance(self, time, other_object, other_time):
        """
        The shortest distance between two objects at specified times.

        Args:
            time (int or datetime): Valid time for this STObject
            other_object: Another STObject being compared
            other_time: The time within the other STObject being evaluated.

        Returns:
            Distance in units of the x-y coordinates
        """
        ti = np.where(self.times == time)[0][0]
        oti = np.where(other_object.times == other_time)[0][0]
        xs = self.x[ti].ravel()[self.masks[ti].ravel() == 1]
        xs = xs.reshape(xs.size, 1)
        ys = self.y[ti].ravel()[self.masks[ti].ravel() == 1]
        ys = ys.reshape(ys.size, 1)
        o_xs = other_object.x[oti].ravel()[other_object.masks[oti].ravel() == 1]
        o_xs = o_xs.reshape(1, o_xs.size)
        o_ys = other_object.y[oti].ravel()[other_object.masks[oti].ravel() == 1]
        o_ys = o_ys.reshape(1, o_ys.size)
        distances = (xs - o_xs) ** 2 + (ys - o_ys) ** 2
        return np.sqrt(distances.min())

    def percentile_distance(self, time, other_object, other_time, percentile):
        ti = np.where(self.times == time)[0][0]
        oti = np.where(other_object.times == other_time)[0][0]
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
        """
        Calculates the center of mass for each time step and outputs an array

        Returns:

        """
        traj = np.zeros((2, self.times.size))
        for t, time in enumerate(self.times):
            traj[:, t] = self.center_of_mass(time)
        return traj

    def get_corner(self, time):
        """
        Gets the corner array indices of the STObject at a given time that corresponds
        to the upper left corner of the bounding box for the STObject.

        Args:
            time: time at which the corner is being extracted.

        Returns:
              corner index.
        """
        if self.start_time <= time <= self.end_time:
            diff = time - self.start_time
            return self.i[diff][0, 0], self.j[diff][0, 0]
        else:
            return -1, -1

    def size(self, time):
        """
        Gets the size of the object at a given time.

        Args:
            time: Time value being queried.

        Returns:
            size of the object in pixels
        """
        if self.start_time <= time <= self.end_time:
            return self.masks[time - self.start_time].sum()
        else:
            return 0

    def max_size(self):
        """
        Gets the largest size of the object over all timesteps.
        
        Returns:
            Maximum size of the object in pixels
        """
        sizes = np.array([m.sum() for m in self.masks])
        return sizes.max()

    def max_intensity(self, time):
        """
        Calculate the maximum intensity found at a timestep.

        """
        ti = np.where(time == self.times)[0][0]
        return self.timesteps[ti].max()

    def extend(self, step):
        """
        Adds the data from another STObject to this object.
        
        Args:
            step: another STObject being added after the current one in time.
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
        Get coordinates of object boundary in counter-clockwise order based on the convex hull of the object.
        For non-convex objects, the convex hull will not be representative of the object shape and boundary_contour
        should be used instead.
        """
        ti = np.where(time == self.times)[0][0]
        com_x, com_y = self.center_of_mass(time)
        # If at least one point along perimeter of the mask rectangle is unmasked, find_boundaries() works.
        # But if all perimeter points are masked, find_boundaries() does not find the object.
        # Therefore, pad the mask with zeroes first and run find_boundaries on the padded array.
        padded_mask = np.pad(self.masks[ti], 1, 'constant', constant_values=0)
        chull = convex_hull_image(padded_mask)
        boundary_image = find_boundaries(chull, mode='inner', background=0)
        # Now remove the padding.
        boundary_image = boundary_image[1:-1, 1:-1]
        boundary_x = self.x[ti].ravel()[boundary_image.ravel()]
        boundary_y = self.y[ti].ravel()[boundary_image.ravel()]
        r = np.sqrt((boundary_x - com_x) ** 2 + (boundary_y - com_y) ** 2)
        theta = np.arctan2((boundary_y - com_y), (boundary_x - com_x)) * 180.0 / np.pi + 360
        polar_coords = np.array([(r[x], theta[x]) for x in range(r.size)], dtype=[('r', 'f4'), ('theta', 'f4')])
        coord_order = np.argsort(polar_coords, order=['theta', 'r'])
        ordered_coords = np.vstack([boundary_x[coord_order], boundary_y[coord_order]])
        return ordered_coords

    def boundary_contour(self, time):
        """
        Calculate the contour around the edge of the binary mask for the object. For objects with interior holes
        or multiple connections, binary dilation, hole filling, and erosion are used to generate a single edge
        contour instead of multiple contours.

        Args:
            time:

        Returns:
            array of shape (2, number of contour points) containing the x and y coordinates of the object edge.
        """
        ti = np.where(time == self.times)[0][0]
        image_mask = binary_erosion(binary_fill_holes(binary_dilation(self.masks[ti])))
        padded_mask = np.pad(image_mask, 1, 'constant', constant_values=0)
        c_out = find_contours(padded_mask, level=0.5, fully_connected="high")
        x_cont = self.x[ti][np.floor(c_out[0][:, 0]).astype(int), np.floor(c_out[0][:, 1]).astype(int)]
        y_cont = self.y[ti][np.floor(c_out[0][:, 0]).astype(int), np.floor(c_out[0][:, 1]).astype(int)]
        ordered_coords = np.vstack([x_cont, y_cont])
        return ordered_coords


    def estimate_motion(self, time, intensity_grid, max_u, max_v):
        """
        Estimate the motion of the object with cross-correlation on the intensity values from the previous time step.

        Args:
            time: time being evaluated.
            intensity_grid: 2D array of intensities used in cross correlation.
            max_u: Maximum x-component of motion. Used to limit search area.
            max_v: Maximum y-component of motion. Used to limit search area

        Returns:
            u, v, and the minimum error.
        """
        ti = np.where(time == self.times)[0][0]
        mask_vals = np.where(self.masks[ti].ravel() == 1)
        i_vals = self.i[ti].ravel()[mask_vals]
        j_vals = self.j[ti].ravel()[mask_vals]
        obj_vals = self.timesteps[ti].ravel()[mask_vals]
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
                # This isn't correlation; it is mean absolute error.
                error = np.abs(shift_vals - obj_vals).mean()
                if error < min_error:
                    min_error = error
                    best_u = u * self.dx
                    best_v = v * self.dx
        # 60 seems arbitrarily high
        # if min_error > 60:
        #    best_u = 0
        #    best_v = 0
        self.u[ti] = best_u
        self.v[ti] = best_v
        return best_u, best_v, min_error

    def count_overlap(self, time, other_object, other_time):
        """
        Counts the number of points that overlap between this STObject and another STObject. Used for tracking.
        """
        ti = np.where(time == self.times)[0][0]
        ma = np.where(self.masks[ti].ravel() == 1)
        oti = np.where(other_time == other_object.times)[0][0]
        obj_coords = np.zeros(np.sum(self.masks[ti]), dtype=[('x', int), ('y', int)])
        other_obj_coords = np.zeros(np.sum(other_object.masks[oti]), dtype=[('x', int), ('y', int)])
        obj_coords['x'] = self.i[ti].ravel()[ma]
        obj_coords['y'] = self.j[ti].ravel()[ma]
        other_obj_coords['x'] = other_object.i[oti][other_object.masks[oti] == 1]
        other_obj_coords['y'] = other_object.j[oti][other_object.masks[oti] == 1]
        return float(np.intersect1d(obj_coords,
                                    other_obj_coords).size) / np.maximum(self.masks[ti].sum(),
                                                                         other_object.masks[oti].sum())

    def extract_attribute_grid(self, model_grid, potential=False, future=False):
        """
        Extracts the data from a ModelOutput or ModelGrid object within the bounding box region of the STObject.
        
        Args:
            model_grid: A ModelGrid or ModelOutput Object
            potential: Extracts from the time before instead of the same time as the object
        """

        if potential:
            var_name = model_grid.variable + "-potential"
            timesteps = np.arange(self.start_time - 1, self.end_time)
        elif future:
            var_name = model_grid.variable + "-future"
            timesteps = np.arange(self.start_time + 1, self.end_time + 2)
        else:
            var_name = model_grid.variable
            timesteps = np.arange(self.start_time, self.end_time + 1)
        self.attributes[var_name] = []
        for ti, t in enumerate(timesteps):
            self.attributes[var_name].append(
                model_grid.data[t - model_grid.start_hour, self.i[ti], self.j[ti]])

    def extract_attribute_array(self, data_array, var_name):
        """
        Extracts data from a 2D array that has the same dimensions as the grid used to identify the object.

        Args:
            data_array: 2D numpy array

        """
        if var_name not in self.attributes.keys():
            self.attributes[var_name] = []
        for t in range(self.times.size):
            self.attributes[var_name].append(data_array[self.i[t], self.j[t]])

    def extract_tendency_grid(self, model_grid):
        """
        Extracts the difference in model outputs

        Args:
            model_grid: ModelOutput or ModelGrid object.

        """
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
        
        Args:
            statistic_name (string): numpy statistic, such as mean, std, max, min

        Returns:
            dict of statistics from each attribute grid.
        """
        stats = {}
        for var, grids in self.attributes.items():
            if len(grids) > 1:
                stats[var] = getattr(np.array([getattr(np.ma.array(x, mask=self.masks[t] == 0), statistic_name)()
                                               for t, x in enumerate(grids)]), statistic_name)()
            else:
                stats[var] = getattr(np.ma.array(grids[0], mask=self.masks[0] == 0), statistic_name)()
        return stats

    def calc_attribute_statistic(self, attribute, statistic, time):
        """
        Calculate statistics based on the values of an attribute. The following statistics are supported:
        mean, max, min, std, ptp (range), median, skew (mean - median), and percentile_(percentile value).

        Args:
            attribute: Attribute extracted from model grid
            statistic: Name of statistic being used.
            time: timestep of the object being investigated

        Returns:
            The value of the statistic
        """
        ti = np.where(self.times == time)[0][0]
        ma = np.where(self.masks[ti].ravel() == 1)
        if len(self.attributes[attribute][ti].ravel()[ma]) < 1:
            stat_val = np.nan
            return stat_val
        if statistic in ['mean', 'max', 'min', 'std', 'ptp']:
            stat_val = getattr(self.attributes[attribute][ti].ravel()[ma], statistic)()
        elif statistic == 'median':
            stat_val = np.median(self.attributes[attribute][ti].ravel()[ma])
        elif statistic == "skew":
            stat_val = np.mean(self.attributes[attribute][ti].ravel()[ma]) - \
                       np.median(self.attributes[attribute][ti].ravel()[ma])
        elif 'percentile' in statistic:
            per = int(statistic.split("_")[1])
            stat_val = np.percentile(self.attributes[attribute][ti].ravel()[ma], per)
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

        Args:
            statistic: statistic being calculated
            time: Timestep being investigated

        Returns:
            Value of the statistic
        """
        ti = np.where(self.times == time)[0][0]
        ma = np.where(self.masks[ti].ravel() == 1)
        if statistic in ['mean', 'max', 'min', 'std', 'ptp']:
            stat_val = getattr(self.timesteps[ti].ravel()[ma], statistic)()
        elif statistic == 'median':
            stat_val = np.median(self.timesteps[ti].ravel()[ma])
        elif 'percentile' in statistic:
            per = int(statistic.split("_")[1])
            stat_val = np.percentile(self.timesteps[ti].ravel()[ma], per)
        elif 'dt' in statistic:
            stat_name = statistic[:-3]
            if ti == 0:
                stat_val = 0
            else:
                stat_val = self.calc_timestep_statistic(stat_name, time) - \
                           self.calc_timestep_statistic(stat_name, time - 1)
        else:
            stat_val = np.nan
        return stat_val

    def calc_shape_statistics(self, stat_names):
        """
        Calculate shape statistics using regionprops applied to the object mask.
        
        Args:
            stat_names: List of statistics to be extracted from those calculated by regionprops.

        Returns:
            Dictionary of shape statistics
        """
        stats = {}
        try:
            all_props = [regionprops(m) for m in self.masks]
        except TypeError:
            raise TypeError("masks not the right type")
        for stat in stat_names:
            stats[stat] = np.mean([p[0][stat] for p in all_props])
        return stats

    def calc_shape_step(self, stat_names, time):
        """
        Calculate shape statistics for a single time step

        Args:
            stat_names: List of shape statistics calculated from region props
            time: Time being investigated

        Returns:
            List of shape statistics

        """
        ti = np.where(self.times == time)[0][0]
        shape_stats = []
        try:
            props = regionprops(self.masks[ti], self.timesteps[ti])[0]
        except:
            for stat_name in stat_names:
                shape_stats.append(np.nan)
            return shape_stats

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

    def to_geojson_feature(self, proj, output_grids=False):
        """
        Output the data in the STObject to a geoJSON file.

        Args:
            proj: PyProj object for converting the x and y coordinates back to latitude and longitude values.
            output_grids: Whether or not to output the primary gridded fields to the geojson file.
        """
        json_features = []
        for t, time in enumerate(self.times):
            feature = {"type": "Feature",
                       "geometry": {"type": "Polygon"},
                       "properties": {}}
            boundary_coords = self.boundary_contour(time)
            lonlat = np.vstack(proj(boundary_coords[0], boundary_coords[1], inverse=True))
            lonlat_list = lonlat.T.tolist()
            feature["geometry"]["coordinates"] = [lonlat_list]
            if output_grids:
                for attr in ["timesteps", "masks", "x", "y", "i", "j"]:
                    feature["properties"][attr] = getattr(self, attr)[t].tolist()
                lon_grid, lat_grid = proj(self.x[t], self.y[t], inverse=True)
                feature["properties"]["lon"] = lon_grid.tolist()
                feature["properties"]["lat"] = lat_grid.tolist()
            if type(time) in [int, np.int32, np.int64]:
                feature["properties"]["valid_time"] = int(time)
            else:
                feature["properties"]["valid_time"] = str(time)
            feature["properties"]["centroid_lon"], \
                feature["properties"]["centroid_lat"] = proj(*self.center_of_mass(time), inverse=True)
            json_features.append(feature)
        return json_features

    def to_geojson(self, filename, proj, metadata=None):
        """
        Output the data in the STObject to a geoJSON file.

        Args:
            filename: Name of the file
            proj: PyProj object for converting the x and y coordinates back to latitude and longitue values.
            metadata: Metadata describing the object to be included in the top-level properties.
        """
        if metadata is None:
            metadata = {}
        json_obj = {"type": "FeatureCollection", "features": [], "properties": {}}
        json_obj['properties']['times'] = self.times.tolist()
        json_obj['properties']['dx'] = self.dx
        json_obj['properties']['step'] = self.step
        json_obj['properties']['u'] = self.u.tolist()
        json_obj['properties']['v'] = self.v.tolist()
        for k, v in metadata.items():
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
            for attr_name, steps in self.attributes.items():
                feature["properties"]["attributes"][attr_name] = steps[t].tolist()
            json_obj['features'].append(feature)
        file_obj = open(filename, "w")
        json.dump(json_obj, file_obj, indent=1, sort_keys=True)
        file_obj.close()
        return


def read_geojson(filename):
    """
    Reads a geojson file containing an STObject and initializes a new STObject from the information in the file.

    Args:
        filename: Name of the geojson file

    Returns:
        an STObject
    """
    json_file = open(filename)
    data = json.load(json_file)
    json_file.close()
    times = data["properties"]["times"]
    main_data = dict(timesteps=[], masks=[], x=[], y=[], i=[], j=[])
    attribute_data = dict()
    for feature in data["features"]:
        for main_name in main_data.keys():
            main_data[main_name].append(np.array(feature["properties"][main_name]))
        for k, v in feature["properties"]["attributes"].items():
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
    for k, v in attribute_data.items():
        sto.attributes[k] = v
    return sto
