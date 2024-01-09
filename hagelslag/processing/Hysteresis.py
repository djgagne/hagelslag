import numpy as np
from scipy.ndimage import label, maximum, find_objects
from numba import jit

class Hysteresis(object):
    """
    Object segmentation method that identifies objects as contiguous areas with all pixels above a low
    threshold and contain at least one pixel above a high threshold.

    Attributes:
        min_intensity: lower threshold value
        max_intensity: higher threshold value
    """

    def __init__(self, min_intensity, max_intensity):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        return

    def label(self, input_grid):
        """
        Label input grid with hysteresis method.

        Args:
            input_grid: 2D array of values.

        Returns:
            Labeled output grid.
        """
        high_labels, num_labels = label(input_grid > self.max_intensity)
        region_ranking = np.argsort(maximum(input_grid, high_labels, index=np.arange(1, num_labels + 1)))[::-1]
        if np.ma.isMA(input_grid):
            input_grid_unmasked = np.where(input_grid.mask, 0, input_grid)
        else:
            input_grid_unmasked = input_grid
        output_grid = grow_objects(input_grid_unmasked, high_labels, region_ranking, self.min_intensity)
        return output_grid

    @staticmethod
    def size_filter(labeled_grid, min_size):
        """
        Remove labeled objects that do not meet size threshold criteria.

        Args:
            labeled_grid: 2D output from label method.
            min_size: minimum size of object in pixels.

        Returns:
            labeled grid with smaller objects removed.
        """
        out_grid = np.zeros(labeled_grid.shape, dtype=np.int64)
        slices = find_objects(labeled_grid)
        j = 1
        for i, s in enumerate(slices):
            box = labeled_grid[s]
            size = np.count_nonzero(box.ravel() == (i + 1))
            if size >= min_size and box.shape[0] > 1 and box.shape[1] > 1:
                out_grid[np.where(labeled_grid == i + 1)] = j
                j += 1
        return out_grid

@jit(nopython=True)
def grow_objects(input_grid, high_labels, region_ranking, min_intensity):
    unset = 0
    output_grid = np.zeros(input_grid.shape, dtype=np.int64)
    stack = []
    for rank in region_ranking:
        label_num = rank + 1
        label_i, label_j = np.where(high_labels == label_num)
        for i in range(label_i.size):
            if output_grid[label_i[i], label_j[i]] == unset:
                stack.append((label_i[i], label_j[i]))
        while len(stack) > 0:
            index = stack.pop(-1)
            output_grid[index] = label_num
            for i in range(index[0] - 1, index[0] + 2):
                for j in range(index[1] - 1, index[1] + 2):
                    if 0 <= i < output_grid.shape[0] and 0 <= j < output_grid.shape[1]:
                        if (input_grid[i, j] > min_intensity) and (output_grid[i, j] == unset):
                            stack.append((i, j))
    return output_grid