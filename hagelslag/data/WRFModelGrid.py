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

    def get_global_attributes(self):
        attributes = {}
        if self.patch_files:
            patch_files = sorted(glob(self.path + "{0}/{0}_".format(self.wrf_filename) + "[0-9]" * 5))
            wrf_data = Dataset(patch_files[0])
        else:
            wrf_data = Dataset(self.wrf_filename)
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
        patch_grid = dict()
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
        return patch_grid, (1,
                            patch_info["BOTTOM-TOP_GRID_DIMENSION"] - 1,
                            patch_info["SOUTH-NORTH_GRID_DIMENSION"] - 1,
                            patch_info["WEST-EAST_GRID_DIMENSION"] - 1)

    def load_full_grid(self):
        var_data = None
        if self.patch_files:
            patch_file_list = sorted(glob(self.path + "{0}/{0}_".format(self.wrf_filename) + "[0-9]" * 5))
            patch_zero = Dataset(patch_file_list[0])
            var_list = patch_zero.variables.keys()
            if self.variable in var_list:
                is_stag = np.array(["stag" in x for x in patch_zero.variables[self.variable].dimensions])
                patch_zero.close()
                if np.any(is_stag):
                    grid_dim_arr = np.array(self.grid_dim)
                    grid_dim_arr[is_stag] += 1
                    var_data = np.zeros(grid_dim_arr, dtype=np.float32)
                    stag_dim = np.where(is_stag)[0][0]
                else:
                    var_data = np.zeros(self.grid_dim, dtype=np.float32)
                    stag_dim = None
                for p, patch_file in enumerate(patch_file_list):
                    wrf_patch_file = Dataset(patch_file)
                    sub_data = wrf_patch_file.variables[self.variable][:]
                    if stag_dim is not None:
                        if len(sub_data.shape) - stag_dim == 1:
                            we_slice = slice(getattr(wrf_patch_file, "WEST-EAST_PATCH_START_STAG") - 1,
                                             getattr(wrf_patch_file, "WEST-EAST_PATCH_END_STAG"))
                            sn_slice = slice(getattr(wrf_patch_file, "SOUTH-NORTH_PATCH_START_UNSTAG") - 1,
                                             getattr(wrf_patch_file, "SOUTH-NORTH_PATCH_END_UNSTAG"))
                        elif len(sub_data.shape) - stag_dim == 2:
                            we_slice = slice(getattr(wrf_patch_file, "WEST-EAST_PATCH_START_UNSTAG") - 1,
                                             getattr(wrf_patch_file, "WEST-EAST_PATCH_END_UNSTAG"))
                            sn_slice = slice(getattr(wrf_patch_file, "SOUTH-NORTH_PATCH_START_STAG") - 1,
                                             getattr(wrf_patch_file, "SOUTH-NORTH_PATCH_END_STAG"))
                        else:
                            we_slice = slice(getattr(wrf_patch_file, "WEST-EAST_PATCH_START_UNSTAG") - 1,
                                             getattr(wrf_patch_file, "WEST-EAST_PATCH_END_UNSTAG"))
                            sn_slice = slice(getattr(wrf_patch_file, "SOUTH-NORTH_PATCH_START_UNSTAG") - 1,
                                             getattr(wrf_patch_file, "SOUTH-NORTH_PATCH_END_UNSTAG"))
                    else:
                        we_slice = slice(getattr(wrf_patch_file, "WEST-EAST_PATCH_START_UNSTAG") - 1,
                                         getattr(wrf_patch_file, "WEST-EAST_PATCH_END_UNSTAG"))
                        sn_slice = slice(getattr(wrf_patch_file, "SOUTH-NORTH_PATCH_START_UNSTAG") - 1,
                                         getattr(wrf_patch_file, "SOUTH-NORTH_PATCH_END_UNSTAG"))
                    if len(var_data.shape) == 3:
                        var_data[:, sn_slice, we_slice] = sub_data
                    else:
                        var_data[:, :, sn_slice, we_slice] = sub_data
                    wrf_patch_file.close()
                if stag_dim is not None:
                    if stag_dim == 1:
                        var_data = 0.5 * (var_data[:, :-1] + var_data[:, 1:])
                    elif stag_dim == 2:
                        var_data = 0.5 * (var_data[:, :, :-1] + var_data[:, :, 1:])
                    else:
                        var_data = 0.5 * (var_data[:, :, :, :-1] + var_data[:, :, :, 1:])
            else:
                patch_zero.close()

        else:
            wrf_data = Dataset(self.path + self.wrf_filename)
            is_stag = np.array(["stag" in x for x in wrf_data.variables[self.variable].dimensions])
            if self.variable in wrf_data.variables.keys():
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
