#!/home/tmp/aburke/miniconda3/bin/python -u
from hagelslag.util.make_proj_grids import read_ncar_map_file
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import os
from netCDF4 import Dataset
from os.path import exists

class SectorProcessor(object):
    def __init__(self,mapfile,
                ensemble_name,member,
                run_date,date_format):

        self.mapfile = mapfile
        self.ensemble_name = ensemble_name
        self.member = member
        self.run_date = run_date
        self.date_format = date_format
        self.inds = None 
        proj_dict, grid_dict = read_ncar_map_file(self.mapfile) 
        
        self.ne_lat, self.sw_lat = grid_dict["ne_lat"],grid_dict["sw_lat"]
        self.ne_lon, self.sw_lon = grid_dict["ne_lon"],grid_dict["sw_lon"]

        return
    
    def output_sector_csv(self,csv_path,file_dict_key,out_path):
        """
        Segment forecast tracks to only output data contined within a 
        region in the CONUS, as defined by the mapfile.

        Args:
            csv_path(str): Path to the full CONUS csv file.
            file_dict_key(str): Dictionary key for the csv files, 
                currently either 'track_step' or 'track_total'
            out_path (str): Path to output new segmented csv files.
        Returns:
            Segmented forecast tracks in a csv file.
        """
        csv_file = csv_path + "{0}_{1}_{2}_{3}.csv".format(
                                                        file_dict_key,
                                                        self.ensemble_name,
                                                        self.member,
                                                        self.run_date.strftime(self.date_format))
        if exists(csv_file):
            csv_data = pd.read_csv(csv_file)
            
            if self.inds is None:
                lon_obj = csv_data.loc[:,"Centroid_Lon"]
                lat_obj = csv_data.loc[:,"Centroid_Lat"]
            
                self.inds = np.where((self.ne_lat>=lat_obj)&(self.sw_lat<=lat_obj)\
                        &(self.ne_lon>=lon_obj)&(self.sw_lon<=lon_obj))[0]
            
            if np.shape(self.inds)[0] > 0:
                csv_data = csv_data.reindex(np.array(self.inds)) 
                sector_csv_filename = out_path + "{0}_{1}_{2}_{3}.csv".format(
                                                        file_dict_key,
                                                        self.ensemble_name,
                                                        self.member,
                                                        self.run_date.strftime(self.date_format))
                print("Output sector csv file " + sector_csv_filename)
                csv_data.to_csv(sector_csv_filename,
                        na_rep="nan",
                        float_format="%0.5f",
                        index=False)
                os.chmod(sector_csv_filename, 0o666)
            else:
                print('No {0} {1} sector data found'.format(self.member,
                                self.run_date.strftime("%Y%m%d")))
            
        else:
            print('No {0} {1} csv file found'.format(self.member,
                                self.run_date.strftime("%Y%m%d")))
        return 


    def load_netcdf_data(self,netcdf_path,patch_radius):
        nc_file = netcdf_path+ "{0}_{1}_{2}_model_patches.nc".format(
                                                            self.ensemble_name,
                                                            self.run_date.strftime(self.date_format),
                                                            self.member)
        if exists(nc_file):
            nc_data = Dataset(nc_file)
            
            lon_obj = nc_data.variables['centroid_lon'][:]
            lat_obj = nc_data.variables['centroid_lat'][:]
            
            inds = np.where((self.ne_lat>=lat_obj)&(self.sw_lat<=lat_obj)\
                        &(self.ne_lon>=lon_obj)&(self.sw_lon<=lon_obj))[0]
            if np.shape(inds)[0] > 0:
                for var in nc_data.variables:
                    if np.shape(nc_data.variables[var])[0] > (2*patch_radius):
                        try:
                            nc_data.variables[var] =\
                            nc_data.variables[var][np.array(inds),:,:]
                        except:
                            nc_data.variables[var] =\
                                nc_data.variables[var][np.array(inds)]
            else:
                nc_data=None
        else:
            nc_data=None
        
        return nc_data
    
    def output_sector_netcdf(self,netcdf_path,out_path,patch_radius,config):
        """
        Segment patches of forecast tracks to only output data contined within a 
        region in the CONUS, as defined by the mapfile.

        Args:
            netcdf_path (str): Path to the full CONUS netcdf patch file.
            out_path (str): Path to output new segmented netcdf files.
            patch_radius (int): Size of the patch radius.
            config (dict): Dictonary containing information about data and
                            ML variables
        Returns:
            Segmented patch netcdf files.
        """
        
        nc_data = self.load_netcdf_data(netcdf_path,patch_radius)
    
        if nc_data is not None:
            out_filename = out_path + "{0}_{1}_{2}_model_patches.nc".format(
                                                            self.ensemble_name,
                                                            self.run_date.strftime(self.date_format),
                                                            self.member)
            out_file = Dataset(out_filename, "w")
            out_file.createDimension("p", np.shape(nc_data.variables['p'])[0])
            out_file.createDimension("row", np.shape(nc_data.variables['row'])[0])
            out_file.createDimension("col", np.shape(nc_data.variables['col'])[0])
            out_file.createVariable("p", "i4", ("p",))
            out_file.createVariable("row", "i4", ("row",))
            out_file.createVariable("col", "i4", ("col",))
            out_file.variables["p"][:] = nc_data.variables['p'][:]
            out_file.variables["row"][:] =  nc_data.variables['row'][:]
            out_file.variables["col"][:] =  nc_data.variables['col'][:]
            out_file.Conventions = "CF-1.6"
            out_file.title = "{0} Storm Patches for run {1} member {2}".format(self.ensemble_name,
                                                                       self.run_date.strftime(self.date_format),
                                                                       self.member)
            out_file.object_variable = config.watershed_variable
            meta_variables = ["lon", "lat", "i", "j", "x", "y", "masks"]
            meta_units = ["degrees_east", "degrees_north", "", "", "m", "m", ""]
            center_vars = ["time", "centroid_lon", "centroid_lat", "centroid_i", "centroid_j", "track_id", "track_step"]
            center_units = ["hours since {0}".format(self.run_date.strftime("%Y-%m-%d %H:%M:%S")),
                    "degrees_east",
                    "degrees_north",
                    "",
                    "",
                    "",
                    ""]

            label_columns = ["Matched", "Max_Hail_Size", "Num_Matches", "Shape", "Location", "Scale"]
        
            for m, meta_variable in enumerate(meta_variables):
                if meta_variable in ["i", "j", "masks"]:
                    dtype = "i4"
                else:
                    dtype = "f4"
                m_var = out_file.createVariable(meta_variable, dtype, ("p", "row", "col"), complevel=1, zlib=True)
                m_var.long_name = meta_variable
                m_var.units = meta_units[m]
        
            for c, center_var in enumerate(center_vars):
                if center_var in ["time", "track_id", "track_step"]:
                    dtype = "i4"
                else:
                    dtype = "f4"
                c_var = out_file.createVariable(center_var, dtype, ("p",), zlib=True, complevel=1)
                c_var.long_name = center_var
                c_var.units =center_units[c]
        
            for storm_variable in config.storm_variables:
                s_var = out_file.createVariable(storm_variable + "_curr", "f4", ("p", "row", "col"), complevel=1, zlib=True)
                s_var.long_name = storm_variable
                s_var.units = ""
        
            for potential_variable in config.potential_variables:
                p_var = out_file.createVariable(potential_variable + "_prev", "f4", ("p", "row", "col"),
                                        complevel=1, zlib=True)
                p_var.long_name = potential_variable
                p_var.units = ""
        
            if config.train:
                for label_column in label_columns:
                    if label_column in ["Matched", "Num_Matches"]:
                        dtype = "i4"
                    else:
                        dtype = "f4"
                    l_var = out_file.createVariable(label_column, dtype, ("p",), zlib=True, complevel=1)
                    l_var.long_name = label_column
                    l_var.units = ""
            
            out_file.variables["time"][:] = nc_data.variables['time'][:]
    
            for c_var in ["lon", "lat"]:
                out_file.variables["centroid_" + c_var][:] =  nc_data.variables['centroid_' + c_var][:]

            for c_var in ["i", "j"]:
                out_file.variables["centroid_" + c_var][:] =  nc_data.variables["centroid_" + c_var][:]
        
            out_file.variables["track_id"][:] =  nc_data.variables['track_id'][:]
            out_file.variables["track_step"][:] =  nc_data.variables['track_step'][:]
        
            for meta_var in meta_variables:
                if meta_var in ["lon", "lat"]:
                    out_file.variables[meta_var][:] = nc_data.variables[meta_var][:]
                else:
                    out_file.variables[meta_var][:] = nc_data.variables[meta_var][:]
        
            for storm_variable in config.storm_variables:
                out_file.variables[storm_variable + "_curr"][:] = nc_data.variables[storm_variable + '_curr'][:]
        
            for p_variable in config.potential_variables:
                out_file.variables[p_variable + "_prev"][:] = nc_data.variables[p_variable + '_prev'][:]
        
            if config.train:
                for label_column in label_columns:
                    try:
                        out_file.variables[label_column][:] = nc_data.variables[label_column][:]
                    except:
                        out_file.variables[label_column][:] = 0

            out_file.close()
        
            print("Output sector nc file " + out_filename)
        else:
            print('No {0} {1} netcdf file/sector data found'.format(self.member,
                                self.run_date.strftime("%Y%m%d")))

        return

if __name__ == "__main__":
    main()
