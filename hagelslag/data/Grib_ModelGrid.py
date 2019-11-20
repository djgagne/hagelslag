#!/usr/bin/env python
import pandas as pd
import pygrib
import numpy as np
from os.path import exists


class Grib_ModelGrid(object):
    """
    Base class for reading 2D model output grids from grib2 files.
    Given a list of file names, loads the values of a single variable from a model run. Supports model output in
    grib2 format
    Attributes:
            filenames (list of str): List of grib2 files containing model output
            run_date (ISO date string or datetime.datetime object): Date of the initialization time of the model run.
            start_date (ISO date string or datetime.datetime object): Date of the first timestep extracted.
            end_date (ISO date string or datetime.datetime object): Date of the last timestep extracted.
            variable (str): Grib2 variable
            member (str): Individual ensemble member.
            frequency (str): Spacing between model time steps.
    """
    def __init__(self,
                 filenames,
                 run_date,
                 start_date,
                 end_date,
                 variable,
                 member,
                 frequency="1H"):

        self.filenames = filenames
        self.variable = variable
        self.run_date = np.datetime64(run_date)
        self.start_date = np.datetime64(start_date)
        self.end_date = np.datetime64(end_date)
        self.frequency = frequency
        self.valid_dates = pd.date_range(start=self.start_date,
                                         end=self.end_date,
                                         freq=self.frequency)
        self.forecast_hours = (self.valid_dates.values - self.run_date).astype("timedelta64[h]").astype(int)
        self.file_objects = []
        self.member = member
        self.__enter__()
        self.data = None
        self.unknown_names = {3: "LCDC", 4: "MCDC", 5: "HCDC", 6: "Convective available potential energy", 7: "Convective inhibition", 
                            197: "RETOP", 198: "MAXREF", 199: "MXUPHL", 200: "MNUPHL", 220: "MAXUVV", 
                            221: "MAXDVV", 222: "MAXUW", 223: "MAXVW"}
        self.unknown_units = {3: "%", 4: "%", 5: "%", 6: "J kg-1", 7: "J kg-1", 197: "m", 198: "dB", 
                            199: "m**2 s**-2", 200: "m**2 s**-2", 220: "m s**-1", 
                            221: "m s**-1", 222: "m s**-1", 223: "m s**-1"}

    def __enter__(self):
        """
        Open each file for reading.
        """
        for filename in self.filenames:
            if exists(filename):
                self.file_objects.append(filename)

    def format_grib_name(self, selected_variable):
        """
        Assigns name to grib2 message number with name 'unknown'. Names based on NOAA grib2 abbreviations.
        
        Names:
            3: LCDC: Low Cloud Cover
            4: MCDC: Medium Cloud Cover
            5: HCDC: High Cloud Cover
            6: Convective available potential energy (CAPE)
            7: Convective Inhibition (CIN)
            197: RETOP: Echo Top
            198: MAXREF: Hourly Maximum of Simulated Reflectivity at 1 km AGL
            199: MXUPHL: Hourly Maximum of Updraft Helicity over Layer 2km to 5 km AGL, and 0km to 3km AGL
                    examples:' MXUPHL_5000' or 'MXUPHL_3000'
            200: MNUPHL: Hourly Minimum of Updraft Helicity at same levels of MXUPHL
                     examples:' MNUPHL_5000' or 'MNUPHL_3000'
            220: MAXUVV: Hourly Maximum of Upward Vertical Velocity in the lowest 400hPa
            221: MAXDVV: Hourly Maximum of Downward Vertical Velocity in the lowest 400hPa
            222: MAXUW: U Component of Hourly Maximum 10m Wind Speed
            223: MAXVW: V Component of Hourly Maximum 10m Wind Speed
        
        Args:
            selected_variable(str): Name of selected variable for loading
        Returns:
            Given an uknown string name of a variable, returns the grib2 message Id
            and units of the variable, based on the self.unknown_name and
            self.unknown_units dictonaries above. Allows access of
            data values of unknown variable name, given the ID.
        """
        names = self.unknown_names
        units = self.unknown_units
        for key, value in names.items():
            if selected_variable == value:
                Id = key
                u = units[key]
        return Id, u

    def load_data(self):
        """
            Loads data from grib2 file objects or list of grib2 file objects. Handles specific grib2 variable names
            and grib2 message numbers.
            Returns:
                    Array of data loaded from files in (time, y, x) dimensions, Units
        """
        if not self.file_objects:
            print()
            print("No {0} model runs on {1}".format(self.member,self.run_date))
            print()
            units = None
            return self.data, units

    
        for f, file in enumerate(self.file_objects):
            grib = pygrib.open(file)
            if type(self.variable) is int:
                data_values = grib[self.variable].values
                if grib[self.variable].units == 'unknown':
                    Id = grib[self.variable].parameterNumber
                    units = self.unknown_units[Id] 
                else:
                    units = grib[self.variable].units
            elif type(self.variable) is str:
                if '_' in self.variable:
                    variable = self.variable.split('_')[0]
                    level = self.variable.split('_')[1]
                    if variable in self.unknown_names.values():
                        Id, units = self.format_grib_name(variable)
                        try:
                            data_values = grib.select(parameterNumber=Id, level=int(level))[0].values
                        except:
                            data_values = grib.select(parameterNumber=Id,typeOfLevel=level)[0].values
                    else:
                        try:
                            data_values = grib.select(name=variable, level=int(level))[0].values
                            units = grib.select(name=variable, level=int(level))[0].units
                        except:
                            data_values = grib.select(name=variable,typeOfLevel=level)[0].values
                            units = grib.select(name=variable, typeOfLevel=level)[0].units
                else:   
                    if self.variable in self.unknown_names.values():
                        Id, units = self.format_grib_name(self.variable)
                        data_values = grib.select(parameterNumber=Id)[0].values
                    elif len(grib.select(name=self.variable)) > 1:
                        raise NameError("Multiple '{0}' records found. Rename with level:'{0}_level'".format(self.variable))
                    else:
                        data_values = grib.select(name=self.variable)[0].values
                        units = grib.select(name=self.variable)[0].units

            if self.data is None:
                self.data = np.empty((len(self.valid_dates), data_values.shape[0], data_values.shape[1]), dtype=float)
                self.data[f]=data_values[:]
            else:
                self.data[f]=data_values[:]
        return self.data, units
    
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
