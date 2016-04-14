from netCDF4 import Dataset
from os.path import isdir, exists
import numpy as np
from glob import glob

class WRFModelGrid(object):
    """
    Load data from raw WRF model output files.
    """
    def __init__(self, forecast_date, variable, domain, path):
        self.forecast_date = forecast_date
        self.variable = variable
        self.domain = domain
        self.path = path
        self.wrf_filename = "wrfout_{0}_{1}".format(self.domain, self.forecast_date.strftime("%Y-%m-%d_%H:%M:%S"))
        self.patch_files = False
        self.patch_grid = None
        self.grid_dim = None
        if exists(self.path + self.wrf_filename):
            if isdir(self.path + self.wrf_filename):
                self.patch_files = True
                self.patch_grid, self.grid_dim = self.calc_patch_grid()
        else:
            raise IOError(self.path + self.wrf_filename + " not found.")
        return

    def calc_patch_grid(self):
        patch_files = sorted(glob(self.path + "{0}/{0}_".format(self.wrf_filename) + "[0-9]" * 5))
        patch_0 = Dataset(patch_files[0])
        patch_attributes = ["WEST-EAST_GRID_DIMENSION", "SOUTH-NORTH_GRID_DIMENSION", "BOTTOM-TOP_GRID_DIMENSION",
                            "WEST-EAST_PATCH_START_UNSTAG", "WEST-EAST_PATCH_END_UNSTAG",
                            "SOUTH-NORTH_PATCH_START_UNSTAG", "SOUTH-NORTH_PATCH_END_UNSTAG"]
        patch_info = {}
        for attribute in patch_attributes:
            patch_info[attribute] = getattr(patch_0, attribute)
        patch_0.close()
        patch_grid = {}
        patch_grid["west_east_start"] = np.arange(0,
                                                  patch_info["WEST-EAST_GRID_DIMENSION"] - 1,
                                                  patch_info["WEST-EAST_PATCH_END_UNSTAG"])

        patch_grid["west_east_end"] = np.arange(patch_info["WEST-EAST_PATCH_END_UNSTAG"],
                                                patch_info["WEST-EAST_GRID_DIMENSION"] + 
                                                patch_info["WEST-EAST_PATCH_END_UNSTAG"],
                                                patch_info["WEST-EAST_PATCH_END_UNSTAG"])
        patch_grid["south_north_start"] = np.arange(0,
                                                    patch_info["SOUTH-NORTH_GRID_DIMENSION"] - 1,
                                                    patch_info["SOUTH-NORTH_PATCH_END_UNSTAG"])
        patch_grid["south_north_end"] = np.arange(patch_info["SOUTH-NORTH_PATCH_END_UNSTAG"],
                                                  patch_info["SOUTH-NORTH_GRID_DIMENSION"] +
                                                  patch_info["SOUTH-NORTH_PATCH_END_UNSTAG"],
                                                  patch_info["SOUTH-NORTH_PATCH_END_UNSTAG"])
        return patch_grid, (patch_info["BOTTOM-TOP_GRID_DIMENSION"] - 1,
                            patch_info["SOUTH-NORTH_GRID_DIMENSION"] - 1,
                            patch_info["WEST-EAST_GRID_DIMENSION"] - 1)

    def load_data_box(self, south_north_slice, west_east_slice):
        """
        Load the data within a specified bounding box.

        Returns:

        """
        if self.patch_files:
            we_patch_start = np.searchsorted(self.patch_grid["west_east_start"], west_east_slice.start) - 1
            we_patch_end = np.searchsorted(self.patch_grid["west_east_end"], west_east_slice.stop) - 1
            sn_patch_start = np.searchsorted(self.patch_grid["south_north_start"], south_north_slice.start) - 1
            sn_patch_end = np.searchsorted(self.patch_grid["south_north_end"], south_north_slice.stop) - 1
            subpatch_rows, subpatch_cols = np.meshgrid(np.arange(sn_patch_start, sn_patch_end + 1),
                                                       np.arange(we_patch_start, we_patch_end + 1))


        else:
            wrf_data = Dataset(self.path + self.wrf_filename)
            if len(wrf_data.variables[self.variable].shape) == 4:
                var_data = wrf_data.variables[self.variable][:, :, south_north_slice, west_east_slice]
            else:
                var_data = wrf_data.variables[self.variable][:, south_north_slice, west_east_slice]
            wrf_data.close()
        return var_data

    def get_slice_patches(self, south_north_slice, west_east_slice):
        return