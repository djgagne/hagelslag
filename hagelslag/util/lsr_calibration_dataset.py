from mpl_toolkits.basemap import Basemap
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
    parser.add_argument("-o", "--out_path", required=True, help="Path to the destination of MESH verification data")
    parser.add_argument("-m", "--map_file", required=True, help="Path to the ensemble mapfile")
    args = parser.parse_args()
    if args.end_date:
        run_dates = pd.DatetimeIndex(start=args.start_date, end=args.end_date, freq='1D').strftime("%y%m%d")
    else:
        run_dates = pd.DatetimeIndex(start=args.start_date, end=args.start_date, freq='1D').strftime("%y%m%d")
    out_path = args.out_path 
    mapfile = args.map_file

    LSR_calibration_data(mapfile,out_path,run_dates)
    return 

def LSR_calibration_data(mapfile,out_path,run_dates,hours=[17,19,21],sector=None):
    """
    Using the grid from input ML forecast (netcdf) data, SPC storm reports 
    with a 25 mile radius around the reports can be plotted. 

    The output file contains binary data, where any point 
    within the 25 mile radius is a 1, and all other points are 0. 

    Currently only supports netcdf files.
    """
    hail_threshold = [25,50]

    lsr_dict = dict()
    
    proj_dict, grid_dict = read_ncar_map_file(mapfile) 
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
    mapping_data = make_proj_grids(proj_dict, grid_dict)
    ML_forecast_lons = mapping_data['lon']
    ML_forecast_lats = mapping_data['lat']
    x1, y1 = m(ML_forecast_lons, ML_forecast_lats)
    forecast_lat = np.array(y1)
    forecast_lon = np.array(x1)
    
    for date in run_dates: 
        print(date)
        csv_file = 'https://www.spc.noaa.gov/climo/reports/{0}_rpts_hail.csv'.format(date)
        try:
            hail_reports = pd.read_csv(csv_file)
        except:
            print('Report CSV file could not be opened.')
            continue           
        for threshold in hail_threshold:
            if os.path.exists(out_path+'{0}_{1}_lsr_mask.nc'.format(date,threshold)):
                print('>{0}mm file already exists'.format(threshold))
                continue
            print('Creating LSR mask >{0}mm'.format(threshold))
            
            #Get size  values from hail reports
            inches_thresh = round((threshold)*0.03937)*100
            report_size = hail_reports.loc[:,'Size'].values
            
            lsr_dict['full_day'] = np.zeros(ML_forecast_lats.shape)
            full_day_indices = np.where(report_size >= inches_thresh)[0]
            if len(full_day_indices) < 1:
                print('No >{0}mm LSRs found'.format(threshold))
                continue 
            reports_lat_full = hail_reports.loc[full_day_indices,'Lat'].values
            reports_lon_full = hail_reports.loc[full_day_indices,'Lon'].values
            lsr_dict['full_day'] = calculate_distance(reports_lat_full,reports_lon_full,forecast_lat,forecast_lon,m)
            
            #Get time  values from hail reports
            report_time = (hail_reports.loc[:,'Time'].values)/100
            #Get lat/lon of different time periods and hail sizes
            for start_hour in hours:
                lsr_dict['{0}'.format(start_hour)] = np.zeros(ML_forecast_lats.shape)
                end_hour = (start_hour+4)%24
                if end_hour > 12:
                    hour_indices = np.where((start_hour <= report_time) & (end_hour >= report_time) & (report_size >= inches_thresh))[0]
                else:
                    #Find reports before and after 0z
                    hour_before_0z = np.where((start_hour <= report_time) & (report_size >= inches_thresh))[0]
                    hour_after_0z = np.where((end_hour >= report_time) & (report_size >= inches_thresh))[0]  
                    #Combine two arrays
                    hour_indices = np.hstack((hour_before_0z, hour_after_0z))
                if len(hour_indices) < 1: 
                    continue
                reports_lat = hail_reports.loc[hour_indices,'Lat'].values
                reports_lon = hail_reports.loc[hour_indices,'Lon'].values
                lsr_dict['{0}'.format(start_hour)] = calculate_distance(reports_lat,reports_lon,forecast_lat,forecast_lon,m)
            
            # Create netCDF file
            if sector:
                out_filename = out_path+'{0}_{1}_{2}_lsr_mask.nc'.format(date,threshold,sector)
            else:
                out_filename = out_path+'{0}_{1}_lsr_mask.nc'.format(date,threshold)
            out_file = Dataset(out_filename, "w")
            out_file.createDimension("x", ML_forecast_lons.shape[0])
            out_file.createDimension("y", ML_forecast_lons.shape[1])
            out_file.createVariable("Longitude", "f4", ("x", "y"))
            out_file.createVariable("Latitude", "f4",("x", "y"))
            out_file.createVariable("24_Hour_All_12z_12z", "f4", ("x", "y"))
            out_file.createVariable("4_Hour_All_17z_21z", "f4", ("x", "y"))
            out_file.createVariable("4_Hour_All_19z_23z", "f4", ("x", "y"))
            out_file.createVariable("4_Hour_All_21z_25z", "f4", ("x", "y"))
            out_file.variables["Longitude"][:,:] = ML_forecast_lons
            out_file.variables["Latitude"][:,:] = ML_forecast_lats
            out_file.variables["24_Hour_All_12z_12z"][:,:] = lsr_dict['full_day']
            out_file.variables["4_Hour_All_17z_21z"][:,:] = lsr_dict['17']
            out_file.variables["4_Hour_All_19z_23z"][:,:] = lsr_dict['19']
            out_file.variables["4_Hour_All_21z_25z"][:,:] = lsr_dict['21']
            out_file.close()
            print("Writing to " + out_filename)
        print() 
    return

def calculate_distance(obs_lat,obs_lon,forecast_lat,forecast_lon,basemap_obj):
    """
    Calculate the difference between forecast data points and observed data.
    Returns:
        Binary array where ones are within a 40km radius 
    """
    x, y = basemap_obj(obs_lon, obs_lat)
    mask_array = np.zeros(forecast_lat.shape)
    for index, point in enumerate(obs_lat):
        lat_diff = (y[index]-forecast_lat)**2.0
        lon_diff = (x[index]-forecast_lon)**2.0
        total_dist = np.sqrt(lat_diff+lon_diff)
        row, col = np.where(total_dist < 40234.0)
        mask_array[row,col] =+ 1.0
    return mask_array

if __name__ == "__main__":
        main()
