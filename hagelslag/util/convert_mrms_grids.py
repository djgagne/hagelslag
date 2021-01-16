import pandas as pd
import os
import subprocess
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree
import pickle
from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta
from hagelslag.util.make_proj_grids import read_arps_map_file, read_ncar_map_file, make_proj_grids
from multiprocessing import Pool
import warnings
import traceback
import argparse
import xarray as xr

def main():
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", required=True, help="Date to begin aggregation in YYYYMMDDHHMM format")
    parser.add_argument("-e", "--end", required=True, help="Date to end aggregation in YYYYMMDDHHMM format")
    parser.add_argument("-p", "--path", required=True, help="Path to MRMS GRIB2 files")
    parser.add_argument("-o", "--out", required=True, help="Path to save MRMS netCDF4 files")
    parser.add_argument("-m", "--map", required=True, help="File containing map coordinates for interpolation")
    parser.add_argument("-v", "--var", required=True,
                        help="Comma-separated list of variables (example: MESH_Max_60min_00.50)")
    parser.add_argument("-i", "--int", default="max", help="Interpolation type ('max' or 'spline')")
    parser.add_argument("-n", "--np", type=int, default=1, help="Number of processors to use")
    args = parser.parse_args()
    start_date = datetime.strptime(args.start, "%Y%m%d%H%M")
    end_date = datetime.strptime(args.end, "%Y%m%d%H%M")
    mrms_path = args.path
    out_path = args.out
    map_filename = args.map
    variables = args.var.split(",")
    num_procs = args.np
    if num_procs > 1:
        pool = Pool(num_procs)
        curr_date = start_date
        while curr_date <= end_date:
            for variable in variables:
                pool.apply_async(interpolate_mrms_day, (curr_date, variable, args.int, mrms_path,
                                                        map_filename, out_path))
            curr_date += timedelta(days=1)
        pool.close()
        pool.join()
    else:
        curr_date = start_date
        while curr_date <= end_date:
            for variable in variables:
                interpolate_mrms_day(curr_date, variable, args.int, mrms_path, map_filename, out_path)
            curr_date += timedelta(days=1)


def load_map_coordinates(map_file):
    """
    Loads map coordinates from netCDF or pickle file created by util.makeMapGrids.

    Args:
        map_file: Filename for the file containing coordinate information.

    Returns:
        Latitude and longitude grids as numpy arrays.
    """
    if map_file[-4:] == ".pkl":
        map_data = pickle.load(open(map_file))
        lon = map_data['lon']
        lat = map_data['lat']
    else:
        map_data = Dataset(map_file)
        if "lon" in map_data.variables.keys():
            lon = map_data.variables['lon'][:]
            lat = map_data.variables['lat'][:]
        else:
            lon = map_data.variables["XLONG"][0]
            lat = map_data.variables["XLAT"][0]
    return lon, lat


def interpolate_mrms_day(start_date, variable, interp_type, mrms_path, map_filename, out_path):
    """
    For a given day, this module interpolates hourly MRMS data to a specified latitude and 
    longitude grid, and saves the interpolated grids to CF-compliant netCDF4 files.
    
    Args:
        start_date (datetime.datetime): Date of data being interpolated
        variable (str): MRMS variable
        interp_type (str): Whether to use maximum neighbor or spline
        mrms_path (str): Path to top-level directory of MRMS GRIB2 files
        map_filename (str): Name of the map filename. Supports ARPS map file format and netCDF files containing latitude
            and longitude variables
        out_path (str): Path to location where interpolated netCDF4 files are saved.
    """
    try:
        print(start_date, variable)
        end_date = start_date + timedelta(hours=23)
        mrms = MRMSGrid(start_date, end_date, variable, mrms_path)
        if mrms.data is not None:
            if map_filename[-3:] == "map":
                mapping_data = make_proj_grids(*read_arps_map_file(map_filename))
                mrms.interpolate_to_netcdf(mapping_data['lon'], mapping_data['lat'], out_path, interp_type=interp_type)
            elif map_filename[-3:] == "txt":
                mapping_data = make_proj_grids(*read_ncar_map_file(map_filename))
                mrms.interpolate_to_netcdf(mapping_data["lon"], mapping_data["lat"], out_path, interp_type=interp_type)
            else:
                lon, lat = load_map_coordinates(map_filename)
                mrms.interpolate_to_netcdf(lon, lat, out_path, interp_type=interp_type)
    except Exception as e:
        # This exception catches any errors when run in multiprocessing, prints the stack trace,
        # and ends the process. Otherwise the process will stall.
        print(traceback.format_exc())
        raise e


class MRMSGrid(object):
    """
    MRMSGrid reads time series of MRMS grib2 files, interpolates them, and outputs them to netCDF4 format.

    """
    def __init__(self, start_date, end_date, variable, path_start, freq="1H"):
        self.start_date = start_date
        self.end_date = end_date
        self.variable = variable
        self.path_start = path_start
        self.freq = freq
        self.data = None
        self.all_dates = pd.date_range(start=self.start_date, end=self.end_date, freq=self.freq)
        self.loaded_dates = None
        self.lon = None
        self.lat = None
        self.load_data()

    def load_data(self):
        """
        Loads data from MRMS GRIB2 files and handles compression duties if files are compressed.
        """
        data = []
        loaded_dates = []
        loaded_indices = []
        for t, timestamp in enumerate(self.all_dates):
            date_str = timestamp.date().strftime("%Y%m%d")
            full_path = self.path_start + date_str + "/"
            if self.variable in os.listdir(full_path):
                full_path += self.variable + "/"
                data_files = sorted(os.listdir(full_path))
                file_dates = pd.to_datetime([d.split("_")[-1][0:13] for d in data_files])
                if timestamp in file_dates:
                    data_file = data_files[np.where(timestamp==file_dates)[0][0]]
                    print(full_path + data_file)
                    if data_file[-2:] == "gz":
                        subprocess.call(["gunzip", full_path + data_file])
                        file_obj = xr.open_dataset(full_path + data_file[:-3])
                    else:
                        file_obj = xr.open_dataset(full_path + data_file)
                    var_name = sorted(file_obj.variables.keys())[0]
                    data.append(file_obj.variables[var_name][:])
                    if self.lon is None:
                        self.lon = file_obj.variables["lon_0"][:]
                        # Translates longitude values from 0:360 to -180:180
                        if np.count_nonzero(self.lon > 180) > 0:
                            self.lon -= 360
                        self.lat = file_obj.variables["lat_0"][:]
                    file_obj.close()
                    if data_file[-2:] == "gz":
                        subprocess.call(["gzip", full_path + data_file[:-3]])
                    else:
                        subprocess.call(["gzip", full_path + data_file])
                    loaded_dates.append(timestamp)
                    loaded_indices.append(t)
        if len(loaded_dates) > 0:
            self.loaded_dates = pd.DatetimeIndex(loaded_dates)
            self.data = np.ones((self.all_dates.shape[0], data[0].shape[0], data[0].shape[1])) * -9999
            self.data[loaded_indices] = np.array(data)

    def interpolate_grid(self, in_lon, in_lat):
        """
        Interpolates MRMS data to a different grid using cubic bivariate splines
        """
        out_data = np.zeros((self.data.shape[0], in_lon.shape[0], in_lon.shape[1]))
        for d in range(self.data.shape[0]):
            print("Loading ", d, self.variable, self.start_date)
            if self.data[d].max() > -999:
                step = self.data[d]
                step[step < 0] = 0
                if self.lat[-1] < self.lat[0]:
                    spline = RectBivariateSpline(self.lat[::-1], self.lon, step[::-1], kx=3, ky=3)
                else:
                    spline = RectBivariateSpline(self.lat, self.lon, step, kx=3, ky=3)
                print("Evaluating", d, self.variable, self.start_date)
                flat_data = spline.ev(in_lat.ravel(), in_lon.ravel())
                out_data[d] = flat_data.reshape(in_lon.shape)
                del spline
            else:
                print(d, " is missing")
                out_data[d] = -9999
        return out_data

    def max_neighbor(self, in_lon, in_lat, radius=0.05):
        """
        Finds the largest value within a given radius of a point on the interpolated grid.

        Args:
            in_lon: 2D array of longitude values
            in_lat: 2D array of latitude values
            radius: radius of influence for largest neighbor search in degrees

        Returns:
            Array of interpolated data
        """
        out_data = np.zeros((self.data.shape[0], in_lon.shape[0], in_lon.shape[1]))
        in_tree = cKDTree(np.vstack((in_lat.ravel(), in_lon.ravel())).T)
        out_indices = np.indices(out_data.shape[1:])
        out_rows = out_indices[0].ravel()
        out_cols = out_indices[1].ravel()
        for d in range(self.data.shape[0]):
            nz_points = np.where(self.data[d] > 0)
            if len(nz_points[0]) > 0:
                nz_vals = self.data[d][nz_points]
                nz_rank = np.argsort(nz_vals)
                original_points = cKDTree(np.vstack((self.lat[nz_points[0][nz_rank]], self.lon[nz_points[1][nz_rank]])).T)
                all_neighbors = original_points.query_ball_tree(in_tree, radius, p=2, eps=0)
                for n, neighbors in enumerate(all_neighbors):
                    if len(neighbors) > 0:
                        out_data[d, out_rows[neighbors], out_cols[neighbors]] = nz_vals[nz_rank][n]
        return out_data

    def interpolate_to_netcdf(self, in_lon, in_lat, out_path, date_unit="seconds since 1970-01-01T00:00",
                              interp_type="spline"):
        """
        Calls the interpolation function and then saves the MRMS data to a netCDF file. It will also create 
        separate directories for each variable if they are not already available.
        """
        if interp_type == "spline":
            out_data = self.interpolate_grid(in_lon, in_lat)
        else:
            out_data = self.max_neighbor(in_lon, in_lat)
        if not os.access(out_path + self.variable, os.R_OK):
            try:
                os.mkdir(out_path + self.variable)
            except OSError:
                print(out_path + self.variable + " already created")
        out_file = out_path + self.variable + "/" + "{0}_{1}_{2}.nc".format(self.variable,
                                                                            self.start_date.strftime("%Y%m%d-%H:%M"),
                                                                            self.end_date.strftime("%Y%m%d-%H:%M"))
        out_obj = Dataset(out_file, "w")
        out_obj.createDimension("time", out_data.shape[0])
        out_obj.createDimension("y", out_data.shape[1])
        out_obj.createDimension("x", out_data.shape[2])
        data_var = out_obj.createVariable(self.variable, "f4", ("time", "y", "x"), zlib=True, 
                                          fill_value=-9999.0,
                                          least_significant_digit=3)
        data_var[:] = out_data
        data_var.long_name = self.variable
        data_var.coordinates = "latitude longitude"
        if "MESH" in self.variable or "QPE" in self.variable:
            data_var.units = "mm"
        elif "Reflectivity" in self.variable:
            data_var.units = "dBZ"
        elif "Rotation" in self.variable:
            data_var.units = "s-1"
        else:
            data_var.units = ""
        out_lon = out_obj.createVariable("longitude", "f4", ("y", "x"), zlib=True)
        out_lon[:] = in_lon
        out_lon.units = "degrees_east"
        out_lat = out_obj.createVariable("latitude", "f4", ("y", "x"), zlib=True)
        out_lat[:] = in_lat
        out_lat.units = "degrees_north"
        dates = out_obj.createVariable("time", "i8", ("time",), zlib=True)
        dates[:] = np.round(date2num(self.all_dates.to_pydatetime(), date_unit)).astype(np.int64)
        dates.long_name = "Valid date"
        dates.units = date_unit
        out_obj.Conventions="CF-1.6"
        out_obj.close()
        return


if __name__ == "__main__":
    main()
