from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from make_proj_grids import *
from netCDF4 import Dataset
import pandas as pd
import numpy as np 
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("-e", "--end_date", required=False, help="End date in YYYY-MM-DD format")
    parser.add_argument("-p", "--data_path", required=True, help="Path to the gridded MESH dataset")
    parser.add_argument("-o", "--out_path", required=True, help="Path to the destination of MESH verification data")
    parser.add_argument("-m", "--map_file", required=True, help="Path to the ensemble map file")
    parser.add_argument("-a", "--mask_file", required=True, help="Path to the ensemble mask file")
    args = parser.parse_args()
    if args.end_date:
        run_dates = pd.DatetimeIndex(start=args.start_date, end=args.end_date, freq='1D').strftime("%y%m%d")
    else:
        run_dates = pd.DatetimeIndex(start=args.start_date, end=args.start_date, freq='1D').strftime("%y%m%d")
    data_path = args.data_path
    out_path = args.out_path 
    mapfile = args.map_file
    maskfile = args.mask_file
    
    MESH_verification_data(data_path,maskfile,mapfile,run_dates,out_path)
    return 

def MESH_verification_data(data_path,maskfile,mapfile,run_dates,out_path,hours=[17,19,21]):
    """
    Calculate 40 km halos around MESH values greater than a threshold value
    """
    hail_threshold = [25,50]

    MESH_dict = dict()
    
    mask_data = Dataset(maskfile)
    mask = mask_data.variables['usa_mask'][:]

    proj_dict, grid_dict = read_ncar_map_file(mapfile) 
    mapping_data = make_proj_grids(proj_dict, grid_dict)
    ML_forecast_lons = mapping_data['lon']
    ML_forecast_lats = mapping_data['lat']
    m = Basemap(projection='lcc', resolution="l",
                rsphere=6371229.0,
                lon_0=proj_dict['lon_0'],
                lat_0=proj_dict['lat_0'],
                lat_1=proj_dict['lat_1'],
                lat_2=proj_dict['lat_2'],
                llcrnrlon=grid_dict['sw_lon'],
                llcrnrlat=grid_dict['sw_lat'],
                urcrnrlon=grid_dict['ne_lon'],
                urcrnrlat=grid_dict['ne_lat'])
    x1, y1 = m(ML_forecast_lons, ML_forecast_lats)
    forecast_lat = np.array(y1)
    forecast_lon = np.array(x1)
    
    for date in run_dates: 
        print(date)
        MESH_file = data_path + 'MESH_Max_60min_00.50_20{0}-00:00_20{0}-23:00.nc'.format(date)
        if not os.path.exists(MESH_file):
            print('No MESH file')
            continue
        MESH_dataset = Dataset(MESH_file)
        MESH_var = MESH_dataset.variables['MESH_Max_60min_00.50']*mask
        MESH_lats = MESH_dataset.variables['latitude'][:]
        MESH_lons = MESH_dataset.variables['longitude'][:]
        
        fullday_MESH = MESH_var[0:23,:,:].max(axis=0)
        
        for threshold in hail_threshold: 
            if os.path.exists(out_path+'{0}_{1}_mesh_mask.nc'.format(date,threshold)):
                print('>{0}mm file already exists'.format(threshold))
                continue
            print('Creating MESH mask >{0}mm'.format(threshold))
            
            MESH_dict['full_day'] = np.zeros(MESH_lons.shape)
            
            thresh_row, thresh_col = np.where(fullday_MESH >= threshold)
            if len(thresh_row)<1:
                print('No >{0}mm MESH values found'.format(threshold))
                continue    
            threshold_MESH_lats = MESH_lats[thresh_row,thresh_col]
            threshold_MESH_lons = MESH_lons[thresh_row,thresh_col]
            MESH_dict['full_day'] = calculate_distance(threshold_MESH_lats,threshold_MESH_lons,forecast_lat,forecast_lon,m)
    
            for hour in hours:
                MESH_dict['{0}'.format(hour)] = np.zeros(MESH_lons.shape)
                
                start_hour = (hour-12)
                end_hour = start_hour+4
                hourly_MESH = MESH_var[start_hour:end_hour,:,:].max(axis=0)
                thresh_row, thresh_col = np.where(hourly_MESH >= threshold)
                if len(thresh_row)<1:
                    continue    
                threshold_MESH_lats = MESH_lats[thresh_row,thresh_col]
                threshold_MESH_lons = MESH_lons[thresh_row,thresh_col]
                MESH_dict['{0}'.format(hour)] = calculate_distance(threshold_MESH_lats,threshold_MESH_lons,forecast_lat,forecast_lon,m)
                
            #Create netcdf file
            out_filename = out_path+'{0}_{1}_mesh_mask.nc'.format(date,threshold)
            out_file = Dataset(out_filename, "w")
            out_file.createDimension("x", MESH_lons.shape[0])
            out_file.createDimension("y", MESH_lons.shape[1])
            out_file.createVariable("Longitude", "f4", ("x", "y"))
            out_file.createVariable("Latitude", "f4",("x", "y"))
            out_file.createVariable("24_Hour_All_12z_12z", "f4", ("x", "y"))
            out_file.createVariable("4_Hour_All_17z_21z", "f4", ("x", "y"))
            out_file.createVariable("4_Hour_All_19z_23z", "f4", ("x", "y"))
            out_file.createVariable("4_Hour_All_21z_25z", "f4", ("x", "y"))
            out_file.variables["Longitude"][:,:] = MESH_lons 
            out_file.variables["Latitude"][:,:] = MESH_lats
            out_file.variables["24_Hour_All_12z_12z"][:,:] = MESH_dict['full_day']
            out_file.variables["4_Hour_All_17z_21z"][:,:] = MESH_dict['17']
            out_file.variables["4_Hour_All_19z_23z"][:,:] = MESH_dict['19']
            out_file.variables["4_Hour_All_21z_25z"][:,:] = MESH_dict['21']
            out_file.close()
            print("Writing to " + out_filename)
        print()
        
    return

def calculate_distance(obs_lat,obs_lon,forecast_lat,forecast_lon,basemap_obj):
    """
    Calculate the difference between forecast data points and observed data.
    Returns:
        Binary array where ones are within a 30km radius 
    """
    x, y = basemap_obj(obs_lon, obs_lat)
    mask_array = np.zeros(forecast_lat.shape)
    for index, point in enumerate(obs_lat):
        lat_diff = (y[index]-forecast_lat)**2.0
        lon_diff = (x[index]-forecast_lon)**2.0
        total_dist = np.sqrt(lat_diff+lon_diff)
        row, col = np.where(total_dist < 30000.0)
        mask_array[row,col] =+ 1.0
    return mask_array

if __name__ == "__main__":
        main()
