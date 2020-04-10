from hagelslag.util.make_proj_grids import make_proj_grids, read_ncar_map_file
from hagelslag.data.ModelOutput import ModelOutput
from hagelslag.data.MRMSGrid import MRMSGrid
from datetime import timedelta
import pandas as pd
import numpy as np
import traceback
import h5py
import os

class DLPreprocessing(object):
    def __init__(self,ensemble_name,model_path,
        hf_path,patch_radius,run_date_format,
        forecast_variables,storm_variables,
        potential_variables,mask=None):
        
        self.ensemble_name = ensemble_name
        self.model_path = model_path
        self.hf_path = hf_path
        self.patch_radius = patch_radius
        self.run_date_format = run_date_format
        self.forecast_variables = forecast_variables
        self.storm_variables = storm_variables
        self.potential_variables = potential_variables
        self.mask = mask
        return

    def process_map_data(self,map_file):
        lon_lat_file = '{0}/{1}_map_data.h5'.format(self.hf_path,self.ensemble_name)
        if not os.path.exists(lon_lat_file):
            proj_dict, grid_dict = read_ncar_map_file(map_file)
            mapping_data = make_proj_grids(proj_dict, grid_dict)
            if self.mask is not None:
                mapping_lat_data = mapping_data['lat']*self.mask
                mapping_lon_data = mapping_data['lon']*self.mask
            else:
                mapping_lat_data = mapping_data['lat']
                mapping_lon_data = mapping_data['lon']
            lon_slices = self.slice_into_patches(mapping_lon_data,self.patch_radius,self.patch_radius)
            lat_slices = self.slice_into_patches(mapping_lat_data,self.patch_radius,self.patch_radius)
            lon_lat_data = np.array((lon_slices,lat_slices))
            print('\nWriting map file: {0}\n'.format(lon_lat_file))
            with h5py.File(lon_lat_file, 'w') as hf:
                hf.create_dataset("map_data",data=lon_lat_data,
                compression='gzip',compression_opts=6)
        return 

    def process_observational_data(self,run_date,start_hour,end_hour,
        mrms_variable, mrms_path):
        """
        Process observational data by both slicing the data and labeling
        MESH values at different thresholds for classification modeling. 
    
        The observational data is in the format (# of hours,x,y)

        Args:
            run_date(datetime): datetime object containing date of mrms data
            config(obj): Config object containing member parameters
        """
        print("Starting obs process", run_date)
        #Find start and end date given the start and end hours 
        start_date = run_date + timedelta(hours=start_hour)
        end_date = run_date + timedelta(hours=end_hour)
        obs_patch_labels = []
        #Create gridded mrms object 
        gridded_obs = MRMSGrid(start_date,end_date,mrms_variable,mrms_path)
        gridded_obs.load_data()
        gridded_obs_data = gridded_obs.data
        if len(gridded_obs_data) < 1: 
            print('No observations on {0}'.format(start_date))
            return
        for hour in range(len(gridded_obs_data[1:])): 
            #Slice mrms data 
            if self.mask is not None: hourly_obs_data = gridded_obs_data[hour]*self.mask
            else: hourly_obs_data = gridded_obs_data[hour]
            hourly_obs_patches = self.slice_into_patches(hourly_obs_data,self.patch_radius,self.patch_radius)
            
            #Label mrms data
            labels = self.label_obs_patches(hourly_obs_patches)
            obs_patch_labels.append(labels)
        obs_filename = '{0}/obs_{1}.h5'.format(self.hf_path,run_date.strftime(self.run_date_format)) 
        print('Writing obs file: {0}'.format(obs_filename))
        
        #Write file out using Hierarchical Data Format 5 (HDF5) format. 
        with h5py.File(obs_filename, 'w') as hf:
            hf.create_dataset("labels",data=obs_patch_labels,
            compression='gzip',compression_opts=6)
        return 

    def process_ensemble_member(self,run_date,member,member_path,map_file,
        start_hour,end_hour,single_step):
        """
        Slice ensemble data in the format (# of hours,x,y)
        Args:
            run_date(datetime): datetime object containing date of mrms data
            member (str): name of the ensemble member
            member_path(str): path to the member patch files 
            lon_lat_file (str): path to the member map file
            config(obj): Config object containing member parameters
        """
        try:
            #Create list of forecast variable strings 
            start_date = run_date + timedelta(hours=start_hour)
            end_date = run_date + timedelta(hours=end_hour)
        
            print("Starting ens processing", member, run_date)
            #Slice each member variable seperately over each hour
            for v,variable in enumerate(self.forecast_variables):
                #Create gridded variable object 
                gridded_variable = ModelOutput(self.ensemble_name,member,run_date,variable,
                        start_date,end_date,self.model_path,map_file,single_step=single_step)
                gridded_variable.load_data() 
                if gridded_variable.data is None: break 
                hourly_var_patches = [] 
                #Slice hourly data
                for hour in np.arange(1,len(gridded_variable.data)):
                    #Storm variables are sliced at the current forecast hour
                    if variable in self.storm_variables:var_hour = hour
                    #Potential (environmental) variables are sliced at the previous forecast hour
                    elif variable in self.potential_variables:var_hour = hour-1
                    if self.mask is not None:masked_gridded_variable = gridded_variable.data[var_hour]*self.mask
                    else:masked_gridded_variable = gridded_variable.data[var_hour]
                    patches = self.slice_into_patches(masked_gridded_variable,self.patch_radius,self.patch_radius)
                    hourly_var_patches.append(patches)
                #Shorten variable names
                if " " in variable: 
                    variable_name= ''.join([v[0].upper() for v in variable.split()]) + variable.split('_')[-1]
                elif "_" in variable: 
                    variable_name= ''.join([v[0].upper() for v in variable.split()]) + variable.split('_')[-1]
                else:variable_name = variable
                var_filename = '{0}/{2}/{1}_{2}_{3}.h5'.format(member_path,
                                                variable_name,
                                                member,
                                                run_date.strftime(self.run_date_format)) 
                print('Writing model file: {0}'.format(var_filename))
                #Write file out using Hierarchical Data Format 5 (HDF5) format. 
                with h5py.File(var_filename, 'w') as hf:
                    hf.create_dataset("patches",data=np.array(hourly_var_patches),
                    compression='gzip',compression_opts=6)
        
        except Exception as e:
            print(traceback.format_exc())
            raise e
    
        return

    def slice_into_patches(self,data2d, patch_ny, patch_nx):
        '''
        A function to slice a 2-dimensional [ny, nx] array into rectangular patches and return 
        the sliced data in an array of shape [npatches, nx_patch, ny_patch].
      
        If the array does not divide evenly into patches, excess points from the northern and 
        eastern edges of the array will be trimmed away (incomplete patches are not included
        in the array returned by this function).

        Input variables:   
                    data2d -- the data you want sliced.  Must be a 2D (nx, ny) array
                    ny_patch -- the number of points in the patch (y-dimension)
                    nx_patch -- the number of points in the patch (x-dimension)
        '''

        #Determine the number of patches in each dimension
        x_patches = int(data2d.shape[0]/patch_nx)
        y_patches = int(data2d.shape[1]/patch_ny) 
        npatches = y_patches * x_patches #Total number of patches
    
        #Define array to store sliced data and populate it from data2d
        sliced_data = [] 
        
        for i in np.arange(0,data2d.shape[0],patch_nx): 
            next_i = i+patch_nx
            if next_i > data2d.shape[0]:
                break 
            for j in  np.arange(0,data2d.shape[1],patch_ny):
                next_j = j+patch_ny
                if next_j > data2d.shape[1]:
                    break
                data = data2d[i:next_i,j:next_j]
                if any(np.isnan(data.flatten())) == True:
                    continue
                sliced_data.append(data)
        return np.array(sliced_data)


    def label_obs_patches(self,obs_patches,label_thresholds=[5,25,50]):
        '''
        A function to generate labels for MESH patch data.  Labels can be defined by passing in a list of
        thresholds on which to divide the categories.  If not provided, default label thresholds of 5, 25, 
        and 50 mm will be used.  The default label thresholds will result in MESH data being labelled as
        follows:
                    Label       Meaning
        No Hail:      0         No pixel exceeding MESH = 5.0 mm in patch 
        Non-severe:   1         5.0 mm < Highest pixel value of MESH in patch < 25.0 mm
        Severe:       2         25.0 mm < Highest pixel value of MESH in patch < 50.0 mm
        Sig. Severe:  3         Highest pixel value of MESH > 50.0 mm

        The input data (obs_patches) must be a list of patches of dimensions [npatches, ny_patch, nx_patch]
        The function returns a list of labels of shape [npatches].

        NOTE:  This function assumes MESH is provided in mm.  If the units of MESH in the input data you
            are using are not mm, either convert them to mm before using this function, or specify 
            appropriate label thresholds using the "label_thresholds" input variable.
        '''
        obs_labels = []
        for k in np.arange(0, obs_patches.shape[0], 1):
            if (np.nanmax(obs_patches[k]) > 50.0):
                label = 3
            elif (np.nanmax(obs_patches[k]) > 25.0):
                label = 2
            elif (np.nanmax(obs_patches[k]) > 5.0):
                label = 1
            else:
                label = 0
            obs_labels.append(label)
        return obs_labels
