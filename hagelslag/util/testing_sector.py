import numpy as np
from create_sector_grid_data import *             
from datetime import datetime
           
from hagelslag.processing.ObjectMatcher import shifted_centroid_distance, start_time_distance
from hagelslag.processing.ObjectMatcher import centroid_distance, time_distance
import os
import pandas as pd
import numpy as np
from datetime import datetime


#date_index = pd.DatetimeIndex([pd.Timestamp.utcnow().strftime("%Y%m%d")])
work_path = "/ai-hail/aburke/2018_HREFv2_data/"
scratch_path= "/hail/aburke/testing_weights/figuring_out_files/"
ensemble_members = ['nam_00','nam_12','arw_00','arw_12','nssl_00','nssl_12','nmmb_00','nmmb_12']

config=dict(dates=None,
            start_hour=12, 
            end_hour=36,
            watershed_variable="MAXUVV",
            ensemble_name="HREFv2",
            ensemble_members=ensemble_members,
            model_path=work_path+'symbolic_hrefv2_data',
            model_watershed_params=(8, 1, 80, 100, 60),
            size_filter=12,
            gaussian_window=2,
            mrms_path=work_path+"MRMS/",
            mrms_variable="MESH_Max_60min_00.50",
            mrms_watershed_params=(19, 1, 100, 100, 75),
            object_matcher_params=([shifted_centroid_distance], np.array([1.0]),
                                     np.array([24000])),
        
            track_matcher_params=([centroid_distance, time_distance],
                                     np.array([80000, 2])),
		
            storm_variables=['MAXUVV','Storm relative helicity_3000', 'Storm relative helicity_1000',
                            'MAXREF','MXUPHL_5000','MAXDVV'], 

            potential_variables=['Precipitable water_0','Temperature_1000','Dew point temperature_1000','Geopotential Height_500','Temperature_500', 
                                'Dew point temperature_500','U component of wind_500','V component of wind_500',
                                'Geopotential Height_700','Temperature_700', 'Dew point temperature_700','U component of wind_700',
                                'V component of wind_700','Geopotential Height_850','Temperature_850', 'Dew point temperature_850',
                                'U component of wind_850','V component of wind_850','MAXUW', 'MAXVW',
                                'Surface lifted index','Convective available potential energy_0','Convective inhibition_0'],
        
            tendency_variables=[],
            shape_variables=["area", "eccentricity", "major_axis_length", "minor_axis_length", "orientation",
                               "extent"],
            variable_statistics=["mean", "max", "min", "std", "skew",
                                   "percentile_10", "percentile_50", "percentile_90"],
            csv_path=scratch_path+"track_data_spring2018_MAXUVV_closest_csv/",
            geojson_path=scratch_path+"track_data_spring2018_MAXUVV_closest_json/",
	        nc_path=scratch_path+"track_data_spring2018_MAXUVV_patch_nc/",
            unique_matches=True,
	        patch_radius=16,
            closest_matches=True,
	        match_steps=True,              
	        train=False,
            single_step=True,
            label_type="gamma",
            model_map_file="/hail/aburke/hagelslag/mapfiles/hrefv2_2018_map.txt",
            mask_file="/hail/aburke/hagelslag/mapfiles/hrefv2_us_mask.nc",
            run_date_format="%Y%m%d-%H%M",
            sector_csv_outpath=scratch_path+'track_data_spring2018_MAXUVV_closest_csv/',
            sector_nc_outpath=scratch_path+'track_data_spring2018_MAXUVV_patch_nc/',
            sector_mapfile="/hail/aburke/hagelslag/mapfiles/hrefv2_sectors/C_2018_map.txt"
            )

sector='E'

date_index = pd.DatetimeIndex(start="2018-05-01T00:00", end="2018-06-02T00:00", freq="1D")

sector_mapfile = "/hail/aburke/hagelslag/mapfiles/hrefv2_sectors/{0}_2018_map.txt".format(sector)
ensemble_name="HREFv2"
member='arw_00'
run_date = datetime(2018,7,1) 
run_date_format="%Y%m%d-%H%M" 
csv_path='/hail/aburke/HREF_Scripts_and_Data/hwt_2018/track_data_spring2018_MAXUVV_closest_csv/'
nc_path='/hail/aburke/HREF_Scripts_and_Data/hwt_2018/track_data_spring2018_MAXUVV_patch_nc/'
sector_csv_outpath="/hail/aburke/testing_weights/sector_trained/{0}/track_data_spring2018_MAXUVV_closest_csv/".format(sector)
sector_nc_outpath="/hail/aburke/testing_weights/sector_trained/{0}/track_data_spring2018_MAXUVV_patch_nc/".format(sector)
patch_radius=16

for run_date in date_index:
    for member in ensemble_members:
        sector = SectorProcessor(sector_mapfile,
                        ensemble_name,member,
                        run_date,run_date_format)

        sector.output_sector_netcdf(nc_path,sector_nc_outpath,patch_radius,config)

        #for key in ['track_step', 'track_total']:
        #    sector.output_sector_csv(csv_path,key,sector_csv_outpath)
