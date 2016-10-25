from netCDF4 import Dataset
from os.path import isdir, exists, join
import numpy as np
from glob import glob
import arrow


class WRFModelGrid(object):
    """
    Load data from raw WRF model output files.
    """
    def __init__(self, forecast_date, variable, domain, path, nocolons=False):
        self.forecast_date = arrow.get(forecast_date)
        self.variable = variable
        self.domain = domain
        self.path = path
        self.wrf_filename = "wrfout_d{0:02d}_{1}".format(self.domain, self.forecast_date.format("YYYY-MM-DD_HH:mm:ss"))
        if nocolons:
            self.wrf_filename = self.wrf_filename.replace(":", "_")
        self.patch_files = False
        if exists(join(self.path, self.wrf_filename)):
            if isdir(join(self.path, self.wrf_filename)):
                self.patch_files = True
        else:
            raise IOError(join(self.path, self.wrf_filename) + " not found.")
        return

    def get_global_attributes(self):
        attributes = {}
        if self.patch_files:
            patch_files = sorted(glob(join(self.path, "{0}/{0}_".format(self.wrf_filename) + "[0-9]" * 5)))
            wrf_data = Dataset(patch_files[0])
        else:
            wrf_data = Dataset(join(self.path, self.wrf_filename))
        for attr in wrf_data.ncattrs():
            attributes[attr] = getattr(wrf_data, attr)
        wrf_data.close()
        return attributes

    def load_time_var(self, time_var="XTIME"):
        var_attrs = dict()
        var_val = 0
        if self.patch_files:
            patch_file_list = sorted(glob(join(self.path, "{0}/{0}_".format(self.wrf_filename) + "[0-9]" * 5)))
            patch_zero = Dataset(patch_file_list[0])
            var_list = patch_zero.variables.keys()
            if time_var in var_list:
                for attr in patch_zero.variables[time_var].ncattrs():
                    var_attrs[attr] = getattr(patch_zero.variables[time_var], attr)
                var_val = patch_zero.variables[time_var][:]        
            patch_zero.close()
        else:
            wrf_data = Dataset(join(self.path, self.wrf_filename))
            var_list = wrf_data.variables.keys()
            if time_var in var_list:
                for attr in wrf_data.variables[time_var].ncattrs():
                    var_attrs[attr] = getattr(wrf_data.variables[time_var], attr)
                var_val = wrf_data.variables[time_var][:]   
            wrf_data.close()
        return var_val, var_attrs 

    def load_full_grid(self):
        var_data = None
        var_attrs = dict()
        if self.patch_files:
            patch_file_list = sorted(glob(join(self.path, "{0}/{0}_".format(self.wrf_filename) + "[0-9]" * 5)))
            patch_zero = Dataset(patch_file_list[0])
            var_list = patch_zero.variables.keys()
            if self.variable in var_list:
                dimension_names = patch_zero.variables[self.variable].dimensions
                is_stag = np.array(["stag" in x for x in patch_zero.variables[self.variable].dimensions])
                for attr in patch_zero.variables[self.variable].ncattrs():
                    if attr == "coordinates":
                        var_attrs[attr] = getattr(patch_zero.variables[self.variable], attr)
                        if "_U" in var_attrs[attr]:
                            var_attrs[attr] = var_attrs[attr].replace("_U", "")
                        elif "_V" in var_attrs[attr]:
                            var_attrs[attr] = var_attrs[attr].replace("_V", "")
                    else:
                        var_attrs[attr] = getattr(patch_zero.variables[self.variable], attr)
                if len(dimension_names) == 4:
                    grid_dim = (1,
                                getattr(patch_zero, "BOTTOM-TOP_GRID_DIMENSION") - 1,
                                getattr(patch_zero, "SOUTH-NORTH_GRID_DIMENSION") - 1,
                                getattr(patch_zero, "WEST-EAST_GRID_DIMENSION") - 1)
                else:
                    grid_dim = (1,
                                getattr(patch_zero, "SOUTH-NORTH_GRID_DIMENSION") - 1,
                                getattr(patch_zero, "WEST-EAST_GRID_DIMENSION") - 1)
                patch_zero.close()
                if np.any(is_stag):
                    grid_dim_arr = np.array(grid_dim)
                    grid_dim_arr[is_stag] += 1
                    var_data = np.zeros(grid_dim_arr, dtype=np.float32)
                    stag_dim = np.where(is_stag)[0][0]
                else:
                    var_data = np.zeros(grid_dim, dtype=np.float32)
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
            wrf_data = Dataset(join(self.path, self.wrf_filename))
            is_stag = np.array(["stag" in x for x in wrf_data.variables[self.variable].dimensions])
            if self.variable in wrf_data.variables.keys():
                for attr in wrf_data.variables[self.variable].ncattrs():
                    if attr == "coordinates":
                        var_attrs[attr] = getattr(wrf_data.variables[self.variable], attr)
                        if "_U" in var_attrs[attr]:
                            var_attrs[attr] = var_attrs[attr].replace("_U", "")
                        elif "_V" in var_attrs[attr]:
                            var_attrs[attr] = var_attrs[attr].replace("_V", "")
                    else:
                        var_attrs[attr] = getattr(wrf_data.variables[self.variable], attr)

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

        return var_data, var_attrs
