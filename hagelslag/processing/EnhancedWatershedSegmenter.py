#
# The Enhanced Watershed code is based on the algorithm from Lakshmanan et al. (2009) and was adapted
# from Java code found at https://github.com/lakshmanok/asgbook/blob/master/src/edu/ou/asgbook/segmentation/EnhancedWatershedSegmenter.java.
#
# References
# Valliappa Lakshmanan, Kurt Hondl, and Robert Rabin, 2009: An Efficient, General-Purpose 
#  Technique for Identifying Storm Cells in Geospatial Images. J. Atmos. Oceanic Technol., 26, 523-537.
"""
@author: David John Gagne (djgagne@ou.edu)
"""

import numpy as np
from scipy.ndimage import label as splabel
from scipy.ndimage import find_objects
from collections import OrderedDict


class EnhancedWatershed(object):
    """
    The enhanced watershed performs image segmentation using a modified version of the traditional watershed technique.
    It includes a size criteria and creates foothills around each object to keep them distinct. The object is used to store
    the quantization and size parameters. It can be used to watershed multiple grids.

    :param min_thresh: minimum pixel value for pixel to be part of a region
    :type min_thresh: int
    :param data_increment: quantization interval. Use 1 if you don't want to quantize
    :type data_increment: int
    :param max_thresh: values greater than maxThresh are treated as the maximum threshold
    :type max_thresh: int
    :param size_threshold_pixels: clusters smaller than this threshold are ignored.
    :type size_threshold_pixels: int
    :param delta: maximum number of data increments the cluster is allowed to range over. Larger d results in clusters over larger scales.
    :type delta: int

    """
    def __init__(self, min_thresh, data_increment, max_thresh, size_threshold_pixels, delta):
        self.min_thresh = min_thresh
        self.data_increment = data_increment
        self.max_thresh = max_thresh
        self.min_size = size_threshold_pixels
        self.delta = delta
        self.max_bin = (self.max_thresh - self.min_thresh) / self.data_increment
        self.UNMARKED = -1
        self.GLOBBED = -3
        self.TOOSMALL = -4

    def label(self, input_grid):
        """
        Labels input grid using enhanced watershed algorithm.
         
        :param input_grid: Grid to be labeled.
        :type input_grid: numpy array
        :rtype: labeled numpy array
        """
        marked = self.find_local_maxima(input_grid)
        marked = np.where(marked >= 0, 1, 0)
        markers = splabel(marked)[0]
        return markers

    @staticmethod
    def size_filter(labeled_grid, min_size):
        """
        Removes labeled objects that are smaller than minSize, and relabels the remaining objects.

        :param labeled_grid: Grid that has been labeled
        """
        out_grid = np.zeros(labeled_grid.shape,dtype=int)
        slices = find_objects(labeled_grid)
        j = 1
        for i, s in enumerate(slices):
            box = labeled_grid[s]
            size = np.count_nonzero(box == i + 1)
            if size >= min_size and box.shape[0] > 1 and box.shape[1] > 1:
                out_grid[np.where(labeled_grid == i+1)] = j
                j += 1
        return out_grid

    def find_local_maxima(self, input_grid):
        """
        Finds the local maxima in the inputGrid and perform region growing to identify objects.

        :param input_grid: Input grid
        """
        pixels, q_data = self.quantize(input_grid)
        centers = OrderedDict()
        for p in pixels.iterkeys():
            centers[p] = []
        marked = np.ones(q_data.shape, dtype=int) * self.UNMARKED
        MIN_INFL = int(np.round(1 + 0.5 * np.sqrt(self.min_size)))
        MAX_INFL = 2 * MIN_INFL
        marked_so_far = []
        for b in sorted(pixels.keys(),reverse=True):
            infl_dist = MIN_INFL + int(np.round(float(b) / self.max_bin * (MAX_INFL - MIN_INFL)))
            for p in pixels[b]:
                if marked[p] == self.UNMARKED:
                    ok = False
                    del marked_so_far[:]
                    for (i, j), v in np.ndenumerate(marked[p[0] - infl_dist:p[0] + infl_dist + 1,
                                                    p[1] - infl_dist:p[1]+ infl_dist + 1]):
                        if v == self.UNMARKED:
                            ok = True
                            marked[i - infl_dist + p[0],j - infl_dist + p[1]] = b
                           
                            marked_so_far.append((i - infl_dist + p[0],j - infl_dist + p[1]))
                        else:
                            ok = False
                            break
                    if ok:
                        centers[b].append(p)
                    else:
                        for m in marked_so_far:
                            marked[m] = self.UNMARKED
        marked[:, :] = self.UNMARKED
        deferred_from_last = []
        deferred_to_next = []
        for delta in range(0, self.delta + 1):
            for b in sorted(centers.keys(), reverse=True):
                bin_lower = b - delta
                deferred_from_last[:] = deferred_to_next[:]
                del deferred_to_next[:]
                foothills = []
                n_centers = len(centers[b])
                tot_centers = n_centers + len(deferred_from_last)
                for i in range(tot_centers):
                    if i < n_centers:
                        center = centers[b][i]
                    else:
                        center = deferred_from_last[i - n_centers]
                    if bin_lower < 0:
                        bin_lower = 0
                    if marked[center] == self.UNMARKED:
                        captured = self.set_maximum(q_data, marked, center, bin_lower, foothills)
                        if not captured:
                            deferred_to_next.append(center)
                self.remove_foothills(q_data, marked, b, bin_lower, centers, foothills)
            del deferred_from_last[:]
            del deferred_to_next[:]
        return marked
            
    def set_maximum(self, q_data, marked, center, bin_lower, foothills):
        """
        Grow a region at a certain bin level and check if the region has reached the maximum size.

        :param q_data:
        :param marked:
        :param center:
        :param bin_lower:
        :param foothills:
        :return:
        """
        as_bin = []
        as_glob = []
        marked_so_far = []
        will_be_considered_again = False
        as_bin.append(center)
        center_data = q_data[center]
        while len(as_bin) > 0:
            p = as_bin.pop(-1)
            if marked[p] != self.UNMARKED:
                continue
            marked[p] = q_data[center]
            marked_so_far.append(p)

            for index,val in np.ndenumerate(marked[p[0] - 1:p[0] + 2, p[1] - 1:p[1] + 2]):
                if val == self.UNMARKED:
                    pixel = (index[0] - 1 + p[0],index[1] - 1 + p[1])
                    p_data = q_data[pixel]
                    if (not will_be_considered_again) and (p_data >= 0) and (p_data < center_data):
                        will_be_considered_again = True
                    if p_data >= bin_lower and (np.abs(center_data - p_data) <= self.delta):
                        as_bin.append(pixel)
                    elif p_data >= 0:
                        as_glob.append(pixel)
        if bin_lower == 0:
            will_be_considered_again = False
        big_enough = len(marked_so_far) >= self.min_size
        if big_enough:
            foothills.append((center, as_glob))
        elif will_be_considered_again:
            for m in marked_so_far:
                marked[m] = self.UNMARKED
            del as_bin[:]
            del as_glob[:]
            del marked_so_far[:]
        return big_enough or (not will_be_considered_again)

    def remove_foothills(self, q_data, marked, bin_num, bin_lower, centers, foothills):
        """
        Mark points determined to be foothills as globbed, so that they are not included in
        future searches. Also searches neighboring points to foothill points to determine
        if they should also be considered foothills.

        :param q_data: Quantized data
        :param marked: Marked
        :param bin_num: Current bin being searched
        :param bin_lower: Next bin being searched
        :param centers: dictionary of local maxima considered to be object centers
        :param foothills: List of foothill points being removed.
        :return:
        """
        hills = []
        for foot in foothills:
            center = foot[0]
            hills[:] = foot[1][:]
            while len(hills) > 0:
                pt = hills.pop(-1)
                marked[pt] = self.GLOBBED
                for s_index, val in np.ndenumerate(marked[pt[0]-1:pt[0]+2,pt[1]-1:pt[1]+2]):
                    index = (s_index[0] - 1 + pt[0], s_index[1] - 1 + pt[1])
                    if val == self.UNMARKED:
                        if (q_data[index] >= 0) and \
                                (q_data[index] < bin_lower) and \
                                ((q_data[index] <= q_data[pt]) or self.is_closest(index, center, centers, bin_num)):
                            hills.append(index)
        del foothills[:] 

    @staticmethod
    def is_closest(point, center, centers, bin_num):
        bin_thresh = bin_num / 2
        p_arr = np.array(point)
        c_arr = np.array(center)
        my_dist = np.sum(np.power(p_arr - c_arr, 2))
        for o_bin in range(bin_thresh, len(centers.keys())):
            for c in centers[o_bin]:
                oc_arr = np.array(c)
                if np.sum(np.power(p_arr - oc_arr, 2)) < my_dist:
                    return False
        return True

    def quantize(self, input_grid):
        """
        Quantize a grid into discrete steps based on input parameters.

        :param input_grid: 2-d array of values
        :type input_grid: numpy.ndarray
        :return: Dictionary of value pointing to pixel locations, and quantized 2-d array of data
        """
        pixels = {}
        for i in range(self.max_bin+1):
            pixels[i] = []

        data = (np.array(input_grid, dtype=int) - self.min_thresh) / self.data_increment
        data[data < 0] = -1
        data[data > self.max_bin] = self.max_bin
        good_points = np.where(data >= 0)
        for g in np.arange(good_points[0].shape[0]):
            pixels[data[(good_points[0][g], good_points[1][g])]].append((good_points[0][g], good_points[1][g]))
        return pixels, data

    @staticmethod
    def is_valid(point, shape):
        return np.all((np.array(point) >= 0) & (np.array(shape) - np.array(point) > 0))