from netCDF4 import Dataset
import numpy as np
from pandas import DatetimeIndex, Timestamp
from os.path import exists

class ModelGrid(object):
    """
    Base class for reading 2D model output grids from netCDF files.

    Given a list of file names, loads the values of a single variable from a model run. Supports model output in
    netCDF format.

    Attributes:
        filenames (list of str): List of netCDF files containing model output
        run_date (ISO date string or datetime.datetime object): Date of the initialization time of the model run.
        start_date (ISO date string or datetime.datetime object): Date of the first timestep extracted.
        end_date (ISO date string or datetime.datetime object): Date of the last timestep extracted.
        freqency (str): spacing between model time steps.
        valid_dates: DatetimeIndex of all model timesteps
        forecast_hours: array of all hours in the forecast
        file_objects (list): List of the file objects for each model time step
    """
    def __init__(self, 
                 filenames, 
                 run_date, 
                 start_date, 
                 end_date,
                 variable,
                 frequency="1H"):
        self.filenames = filenames
        self.variable = variable
        self.run_date = np.datetime64(run_date)
        self.start_date = np.datetime64(start_date)
        self.end_date = np.datetime64(end_date)
        self.frequency = frequency
        self.valid_dates = DatetimeIndex(start=self.start_date,
                                         end=self.end_date,
                                         freq=self.frequency)
        self.forecast_hours = (self.valid_dates.values - self.run_date).astype("timedelta64[h]").astype(int)
        self.file_objects = []
        self.__enter__()

    def __enter__(self):
        """
        Open each file for reading.

        """
        for filename in self.filenames:
            if exists(filename):
                self.file_objects.append(Dataset(filename))
            else:
                self.file_objects.append(None)

    def load_data_old(self):
        """
        Loads time series of 2D data grids from each opened file. The code 
        handles loading a full time series from one file or individual time steps
        from multiple files. Missing files are supported.
        """
        units = ""
        if len(self.file_objects) == 1 and self.file_objects[0] is not None:
            data = self.file_objects[0].variables[self.variable][self.forecast_hours]
            if hasattr(self.file_objects[0].variables[self.variable], "units"):
                units = self.file_objects[0].variables[self.variable].units
        elif len(self.file_objects) > 1:
            grid_shape = [len(self.file_objects), 1, 1]
            for file_object in self.file_objects:
                if file_object is not None:
                    if self.variable in file_object.variables.keys():
                        grid_shape = file_object.variables[self.variable].shape
                    elif self.variable.ljust(6, "_") in file_object.variables.keys():
                        grid_shape = file_object.variables[self.variable.ljust(6, "_")].shape

                    else:
                        print("{0} not found".format(self.variable))
                        raise KeyError
                    break
            data = np.zeros((len(self.file_objects), grid_shape[1], grid_shape[2]))
            for f, file_object in enumerate(self.file_objects):
                if file_object is not None:
                    if self.variable in file_object.variables.keys():
                        var_name = self.variable
                    elif self.variable.ljust(6, "_") in file_object.variables.keys():
                        var_name = self.variable.ljust(6, "_")
                    else:
                        print("{0} not found".format(self.variable))
                        raise KeyError
                    data[f] = file_object.variables[var_name][0]
                    if units == "" and hasattr(file_object.variables[var_name], "units"):
                        units = file_object.variables[var_name].units
        else:
            data = None
        return data, units

    def load_data(self):
        """
        Load data from netCDF file objects or list of netCDF file objects. Handles special variable name formats.

        Returns:
            Array of data loaded from files in (time, y, x) dimensions, Units
        """
        units = ""
        if self.file_objects[0] is None:
            raise IOError()
        var_name, z_index = self.format_var_name(self.variable, list(self.file_objects[0].variables.keys()))
        ntimes = 0
        if 'time' in self.file_objects[0].variables[var_name].dimensions:
            ntimes = len(self.file_objects[0].dimensions['time'])

        if ntimes > 1:
            if z_index is None:
                data = self.file_objects[0].variables[var_name][self.forecast_hours].astype(np.float32)
            else:
                data = self.file_objects[0].variables[var_name][self.forecast_hours, z_index].astype(np.float32)
        else:
            y_dim, x_dim = self.file_objects[0].variables[var_name].shape[-2:]
            data = np.zeros((len(self.valid_dates), y_dim, x_dim), dtype=np.float32)
            for f, file_object in enumerate(self.file_objects):
                if file_object is not None:
                    if z_index is None:
                        data[f] = file_object.variables[var_name][0]
                    else:
                        data[f] = file_object.variables[var_name][0, z_index]
        if hasattr(self.file_objects[0].variables[var_name], "units"):
            units = self.file_objects[0].variables[var_name].units
        return data, units

    @staticmethod
    def format_var_name(variable, var_list):
        """
        Searches var list for variable name, checks other variable name format options.

        Args:
            variable (str): Variable being loaded
            var_list (list): List of variables in file.

        Returns:
            Name of variable in file containing relevant data, and index of variable z-level if multiple variables
            contained in same array in file.
        """
        z_index = None
        if variable in var_list:
            var_name = variable
        elif variable.ljust(6, "_") in var_list:
            var_name = variable.ljust(6, "_")
        elif any([variable in v_sub.split("_") for v_sub in var_list]):
            var_name = var_list[[variable in v_sub.split("_") for v_sub in var_list].index(True)]
            z_index = var_name.split("_").index(variable)
        else:
            raise KeyError("{0} not found in {1}".format(variable, var_list))
        return var_name, z_index

    def __exit__(self):
        """
        Close links to all open file objects and delete the objects.
        """
        for file_object in self.file_objects:
            file_object.close()
        del self.file_objects[:]

    def close(self):
        """
        Close links to all open file objects and delete the objects.
        """
        self.__exit__()

