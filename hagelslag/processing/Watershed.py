from skimage.morphology import watershed
from scipy.ndimage import label, find_objects
import numpy as np


class Watershed(object):
    """
    This watershed approach performs a standard labeling of intense objects then grows the intense
    objects out to the minimum intensity. It will create separate objects for the area around each
    core in a line of storms, for example.

    Args:
        min_intensity: minimum intensity for the storm field
        core_intensity: the intensity used to determine the initial objects.

    """
    def __init__(self, min_intensity, max_intensity):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def label(self, data):
        core_labels, n_labels = label(data >= self.max_intensity)
        ws_labels = watershed(data.max() - data, markers=core_labels, mask=data >= self.min_intensity)
        return ws_labels

    @staticmethod
    def size_filter(labeled_grid, min_size):
        """
        Removes labeled objects that are smaller than min_size, and relabels the remaining objects.

        Args:
            labeled_grid: Grid that has been labeled
            min_size: Minimium object size.
        Returns:
            Labeled array with re-numbered objects to account for those that have been removed
        """
        out_grid = np.zeros(labeled_grid.shape, dtype=int)
        slices = find_objects(labeled_grid)
        j = 1
        for i, s in enumerate(slices):
            box = labeled_grid[s]
            size = np.count_nonzero(box == i + 1)
            if size >= min_size and box.shape[0] > 1 and box.shape[1] > 1:
                out_grid[np.where(labeled_grid == i + 1)] = j
                j += 1
        return out_grid
