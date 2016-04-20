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

    def get_dimensions(self, filename):
        dimensions = {}
        wrf_data = Dataset(filename)
        for dim in wrf_data.dimensions.keys():
            dimensions[dim] = wrf_data.dimensions[dim].size
        wrf_data.close()
        return dimensions

    def get_global_attributes(self, filename):
        attributes = {}
        wrf_data = Dataset(filename)
        for attr in wrf_data.ncattrs():
            attributes[attr] = getattr(wrf_data, attr)
        wrf_data.close()
        return attributes

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
        var_data = None
        if self.patch_files:
            we_patch_start = np.searchsorted(self.patch_grid["west_east_start"], west_east_slice.start)[0] - 1
            we_patch_end = np.searchsorted(self.patch_grid["west_east_end"], west_east_slice.stop)[0] - 1
            sn_patch_start = np.searchsorted(self.patch_grid["south_north_start"], south_north_slice.start)[0] - 1
            sn_patch_end = np.searchsorted(self.patch_grid["south_north_end"], south_north_slice.stop)[0] - 1
            subpatch_rows, subpatch_cols = np.meshgrid(np.arange(sn_patch_start, sn_patch_end + 1),
                                                       np.arange(we_patch_start, we_patch_end + 1))
            for patch_row, patch_col in zip(subpatch_rows.ravel(), subpatch_cols.ravel()):
                patch_num = np.ravel_multi_index((patch_row, patch_col),
                                                 (self.patch_grid["south_north_start"].size,
                                                  self.patch_grid["west_east_start"].size))
                patch_data = Dataset(self.path + "{0}/{0}_{1:03d}".format(self.wrf_filename, patch_num))
                patch_data.variables[self.variable]
        else:
            wrf_data = Dataset(self.path + self.wrf_filename)
            if self.variable in wrf_data.variables.keys():
                is_stag = np.array(["stag" in x for x in wrf_data.variables[self.variable].dimensions])
                if np.any(is_stag):
                    stag_dim = np.where(is_stag)[0][0]
                    if stag_dim == 1:
                        stag_data = wrf_data.variables[self.variable][:, :, south_north_slice, west_east_slice]
                        var_data = 0.5 * (stag_data[:, :-1] + stag_data[:, 1:])
                    elif stag_dim == 2:
                        stag_data = wrf_data.variables[self.variable][:, :,
                                    south_north_slice.start:
                                    south_north_slice.stop + 1,
                                    west_east_slice]
                        var_data = 0.5 * (stag_data[:, :, :-1] + stag_data[:, :, 1:])
                    else:
                        stag_data = wrf_data.variables[self.variable][:, :,
                                    south_north_slice,
                                    west_east_slice.start:west_east_slice.stop + 1]
                        var_data = 0.5 * (stag_data[:, :, :, :-1] + stag_data[:, :, :, 1:])
                if len(wrf_data.variables[self.variable].shape) == 4:
                    var_data = wrf_data.variables[self.variable][:, :, south_north_slice, west_east_slice]
                else:
                    var_data = wrf_data.variables[self.variable][:, south_north_slice, west_east_slice]
            wrf_data.close()
        return var_data

    def load_full_grid(self):
        var_data = None
        if self.patch_files:
            patch_files = sorted(glob(self.path + "{0}/{0}_".format(self.wrf_filename) + "[0-9]" * 5))
            for patch_file in patch_files:
                wrf_patch_file = Dataset(patch_file)
                wrf_patch_file.close()
        else:
            wrf_data = Dataset(self.path + self.wrf_filename)
            is_stag = np.array(["stag" in x for x in wrf_data.variables[self.variable].dimensions])
            var_data = wrf_data.variables[self.variable][:]
            if np.any(is_stag):
                stag_dim = np.where(is_stag)[0][0]
                if stag_dim == 1:
                    var_data = 0.5 * (var_data[:, :-1] + var_data[:, 1:])
                elif stag_dim == 2:
                    var_data = 0.5 * (var_data[:, :, :-1] + var_data[:, :, 1:])
                else:
                    var_data = 0.5 * (var_data[:, :, :, :-1] + var_data[:, :, :, 1:])
            wrf_data.close()

        return var_data

