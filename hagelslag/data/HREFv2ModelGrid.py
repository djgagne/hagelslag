#!/usr/bin/env python
import os
from pyproj import Proj
from scipy.spatial import cKDTree

import pygrib
import numpy as np
from os.path import exists
import pandas as pd
from pandas import DatetimeIndex


class ModelGrid(object):
    """
    Base class for reading 2D model output grids from grib2 files.
    Given a list of file names, loads the values of a single variable from a model run. Supports model output in
    grib2 format
    Attributes:
            filenames (list of str): List of grib2 files containing model output
            run_date (ISO date string or datetime.datetime object): Date of the initialization time of the model run.
            start_date (ISO date string or datetime.datetime object): Date of the first timestep extracted.
            end_date (ISO date string or datetime.datetime object): Date of the last timestep extracted.
            freqency (str): spacing between model time steps.
            valid_dates(dattime.datetime): DatetimeIndex of all model timesteps
            forecast_hours(array): array of all hours in the forecast
            file_objects (list): List of the file objects for each model time step
    """

    def __init__(self, filenames,
                 run_date,
                 start_date,
                 end_date,
                 variable,
                 member,
                 mapping_data,
                 sector_ind_path,
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
        self.member = member
        self.mapping_data = mapping_data
        self.sector_ind_path = sector_ind_path
        self.__enter__()
        self.data = None
        self.lat = None
        self.lon = None
        self.unknown_names = {3: "LCDC", 4: "MCDC", 5: "HCDC", 197: "RETOP", 198: "MAXREF", 199: "MXUPHL",
                              200: "MNUPHL", 220: "MAXUVV", 221: "MAXDVV", 222: "MAXUW", 223: "MAXVW"}
        self.unknown_units = {3: "%", 4: "%", 5: "%", 197: "m", 198: "dB", 199: "m**2 s**-2", 200: "m**2 s**-2",
                              220: "m s**-1", 221: "m s**-1", 222: "m s**-1", 223: "m s**-1"}

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
        Args:
            selected_variable(str): name of selected variable for loading
        Names:
            3: LCDC: Low Cloud Cover
            4: MCDC: Medium Cloud Cover
            5: HCDC: High Cloud Cover
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
        file_objects = self.file_objects
        var = self.variable
        valid_date = self.valid_dates
        data = self.data
        unknown_names = self.unknown_names
        unknown_units = self.unknown_units
        member = self.member
        lat = self.lat
        lon = self.lon
      
        if self.sector_ind_path:
            inds_file = pd.read_csv(self.sector_ind_path+'sector_data_indices.csv') 
            inds = inds_file.loc[:,'indices']  
        out_x = self.mapping_data["x"]
        
        if not file_objects:
            print()
            print("No {0} model runs on {1}".format(member,self.run_date))
            print()
            units = None
            return self.data, units

    
        for f, file in enumerate(file_objects):
            grib = pygrib.open(file)
            if type(var) is int:
                data_values = grib[var].values
                #lat, lon = grib[var].latlons()
                #proj = Proj(grib[var].projparams)
                if grib[var].units == 'unknown':
                    Id = grib[var].parameterNumber
                    units = self.unknown_units[Id] 
                else:
                    units = grib[var].units
            elif type(var) is str:
                if '_' in var:
                    variable = var.split('_')[0]
                    level = int(var.split('_')[1])
                    if variable in unknown_names.values():
                        Id, units = self.format_grib_name(variable)
                        data_values = grib.select(parameterNumber=Id, level=level)[0].values
                        #lat, lon =  grib.select(parameterNumber=Id, level=level)[0].latlons()
                        #proj = Proj(grib.select(parameterNumber=Id, level=level)[0].projparams)

                    else:
                        data_values = grib.select(name=variable, level=level)[0].values
                        units = grib.select(name=variable, level=level)[0].units
                        #lat, lon  = grib.select(name=variable, level=level)[0].latlons()
                        #proj = Proj(grib.select(name=variable, level=level)[0].projparams)
                else:
                    if var in unknown_names.values():
                        Id, units = self.format_grib_name(var)
                        data_values = grib.select(parameterNumber=Id)[0].values
                        #lat, lon = grib.select(parameterNumber=Id)[0].latlons() 
                        #proj = Proj(grib.select(parameterNumber=Id)[0].projparams)

                    elif len(grib.select(name=var)) > 1:
                        raise NameError("Multiple '{0}' records found. Rename with level:'{0}_level'".format(var))

                    else:
                        data_values = grib.select(name=var)[0].values
                        units = grib.select(name=var)[0].units
                        #lat, lon = grib.select(name=var)[0].latlons()
                        #proj = Proj(grib.select(name=var)[0].projparams)

            if data is None:
                data = np.empty((len(valid_date), out_x.shape[0], out_x.shape[1]), dtype=float)
                if self.sector_ind_path:
                    data[f] = data_values[:].flatten()[inds].reshape(out_x.shape)
                else:
                    data[f]=data_values[:]
            else:
                if self.sector_ind_path:
                    data[f] = data_values[:].flatten()[inds].reshape(out_x.shape)
                else:
                    data[f]=data_values[:]
        
        return data, units
        
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


class HREFv2ModelGrid(ModelGrid):
    """
    Extension of the ModelGrid class for interfacing with the HREFv2  ensemble.
    Args:
        member (str): Name of the ensemble member
        run_date (datetime.datetime object): Date of the initial step of the ensemble run
        variable(int or str): name of grib2 variable(str) or grib2 message number(int) being loaded
        start_date (datetime.datetime object): First time step extracted.
        end_date (datetime.datetime object): Last time step extracted.
        path (str): Path to model output files
        single_step (boolean (default=True): Whether variable information is stored with each time step in a separate
                file (True) or one file containing all timesteps (False).
    """

    def __init__(self, member, run_date, variable, start_date, 
                end_date, path,mapping_data,sector_ind_path,single_step=True):
        self.path = path
        self.member = member
        filenames = []
        self.forecast_hours = np.arange((start_date - run_date).total_seconds() / 3600,
                                        (end_date - run_date).total_seconds() / 3600 + 1, dtype=int)

        if 'nam' in self.member:
            if '00' in self.member:
                for forecast_hr in self.forecast_hours:
                    file = '{1}/{2}/{3}/nam.t00z.conusnest.camfld{0:02}.tm00.grib2'.format(forecast_hr,
                                                                                           self.path,
                                                                                           self.member,
                                                                                           run_date.strftime("%Y%m%d"))
                    filenames.append(file)
            elif '12' in self.member:
                for forecast_hr in self.forecast_hours:
                    file = '{1}/{2}/{3}/nam.t12z.conusnest.camfld{0:02}.tm00.grib2'.format(forecast_hr,
                                                                                           self.path,
                                                                                           self.member,
                                                                                           run_date.strftime("%Y%m%d"))
                    filenames.append(file)
        else:
            member_name = str(self.member.split("_")[0])
            if '00' in self.member:
                for forecast_hr in self.forecast_hours:
                    file = '{1}/{2}/{3}/hiresw_conus{4}_{3}00f0{0:02}.grib2'.format(forecast_hr,
                                                                                    self.path,
                                                                                    self.member,
                                                                                    run_date.strftime("%Y%m%d"),
                                                                                    member_name)
                    filenames.append(file)

            elif '12' in self.member:
                for forecast_hr in self.forecast_hours:
                    file = '{1}/{2}/{3}/hiresw_conus{4}_{3}12f0{0:02}.grib2'.format(forecast_hr,
                                                                                    self.path,
                                                                                    self.member,
                                                                                    run_date.strftime("%Y%m%d"),
                                                                                    member_name)
                    filenames.append(file)

        super(HREFv2ModelGrid, self).__init__(filenames, run_date, start_date, end_date, 
                                            variable, member,mapping_data,sector_ind_path)
        return
