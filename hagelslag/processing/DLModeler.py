from hagelslag.util.make_proj_grids import make_proj_grids, read_ncar_map_file
from scipy.spatial import cKDTree
from datetime import timedelta
from glob import glob
import pandas as pd
import numpy as np
import random
import h5py
import os

from sklearn.utils import shuffle

from dask.distributed import Client, progress
import dask.array as da
import dask

#from keras.optimizers import SGD, Adam
#from keras.regularizers import l2
#import keras.backend as K
#from keras import layers
#from keras import models

class DLModeler(object):
    def __init__(self,model_path,hf_path,start_dates,end_dates,
        num_examples,class_percentage,patch_radius,run_date_format,forecast_variables):
        
        self.model_path = model_path
        self.hf_path = hf_path
        self.start_dates = start_dates
        self.end_dates = end_dates
        self.num_examples = num_examples
        self.class_percentage = class_percentage
        self.patch_radius = patch_radius
        self.run_date_format = run_date_format
        self.forecast_variables = forecast_variables
        return

    def pre_process_training_data(self,member):
        """
        Function that reads and extracts 2D sliced training data from 
        an ensemble member. 

        Args:
            member (str): Ensemble member data that trains a CNN
        Returns:
            Sliced ensemble member data and observations.
        """
        
        print()
        mode = 'train'
        #Find/create files containing information about training
        filename_args = (member,self.start_dates[mode].strftime('%Y%m%d'),
                        self.end_dates[mode].strftime('%Y%m%d'),self.num_examples)
        #Random patches for training
        training_filename=self.model_path+\
            '/{0}_{1}_{2}_{3}_training_examples.csv'.format(*filename_args)
        #Observations associated with random patches    
        obs_training_labels_filename=self.model_path+\
            '/{0}_{1}_{2}_{3}_training_obs_label.h5'.format(*filename_args)
        #Ensemble member data associated with random patches
        model_training_data_filename = self.model_path+\
            '/{0}_{1}_{2}_{3}_training_patches.h5'.format(*filename_args)
        #Selecting random patches for training
        if os.path.exists(training_filename):
            print('Opening {0}'.format(training_filename))
            patches = pd.read_csv(training_filename,index_col=0)
        else:
            patches = self.training_data_selection(member,training_filename)
        #Creating label data of shape (# examples, #classes)
        if os.path.exists(obs_training_labels_filename):
            print('Opening {0}'.format(obs_training_labels_filename))
            with h5py.File(obs_training_labels_filename, 'r') as hf:
                member_obs_label = hf['data'][()]
        else:
            unique_obs_label = np.unique(patches['Obs Label'])
            member_obs_label = np.zeros( (len(patches['Obs Label']),len(unique_obs_label)) )
            for l, label in enumerate(patches['Obs Label']):
                for u,unique in enumerate(unique_obs_label):
                    if label == unique:
                        member_obs_label[l,u] = 1.0 
            print('Writing out {0}'.format(obs_training_labels_filename))
            with h5py.File(obs_training_labels_filename, 'w') as hf:
                hf.create_dataset("data",data=member_obs_label,
                compression='gzip',compression_opts=6)
        #Extracting model patch data
        if os.path.exists(model_training_data_filename):
            with h5py.File(model_training_labels_filename, 'r') as hf:
                member_model_data = hf['data'][()]
        else:
            member_model_data = self.read_files(mode,member,patches['Random Date'], 
                patches['Random Hour'],patches['Random Patch'],patches['Data Augmentation'])
            if member_model_data is None: print('No training data found')
            else:
                print('Writing out {0}'.format(model_training_data_filename))
                with h5py.File(model_training_data_filename, 'w') as hf:
                    hf.create_dataset("data",data=member_model_data,
                    compression='gzip',compression_opts=6)
        
        return member_model_data,member_obs_label
    


    def training_data_selection(self,member,training_filename):
        """
        Function to select random patches to train a CNN. Observation files
        are loaded and read to create a balanced dataset with 
        multiple class examples.

        Args:
            member (str): Ensemble member data that trains a CNN
            training_filename (str): Filename and path to save the random
                training patches.
        Returns:
            Pandas dataframe with random patch information
        """ 
        string_dates = pd.date_range(start=self.start_dates['train'],
                    end=self.end_dates['train'],freq='1D').strftime(self.run_date_format)
        
        #Place all obs data into respective category
        all_date_obs_catetories = {}
        #Loop through each category:
        for category in [0,1,2,3]:
            single_date_obs = {}
            #Loop through each date
            for str_date in string_dates:
                #If there are model or obs files, continue to next date
                model_file = glob(self.hf_path + '/{0}/*{1}*'.format(member,str_date))
                obs_file = glob(self.hf_path + '/*obs*{0}*'.format(str_date))
                if len(model_file) < 1 or len(obs_file) < 1:continue
                #Open obs file
                data = h5py.File(obs_file[0], 'r')['labels'][()]
                hourly_data = {}
                for hour in np.arange(data.shape[0]):
                    inds = np.where(data[hour] == category)[0]
                    if len(inds) >1: hourly_data[hour] = inds 
                if hourly_data: single_date_obs[str_date] = hourly_data
            if single_date_obs: all_date_obs_catetories[category] = single_date_obs
        
        #Find the number of desired examples per category
        num_examples_obs_categories = [] 
        for class_label,percentage in self.class_percentage.items():
            subset_class_examples = int(self.num_examples*percentage)
            total_class_examples = all_date_obs_catetories[class_label]
            if len(total_class_examples) < subset_class_examples:data_augment = 1
            else:data_augment = 0
            for e in np.arange(subset_class_examples):
                #Choose random dates, hours, and patches for training
                random_date = np.random.choice(list(total_class_examples))
                random_hour = np.random.choice(list(total_class_examples[random_date]))
                random_patch = np.random.choice(list(total_class_examples[random_date][random_hour]))
                num_examples_obs_categories.append([random_date,random_hour,random_patch,class_label,data_augment])
        
        #Output dataframe with the randomly chosen data
        cols = ['Random Date','Random Hour', 'Random Patch', 'Obs Label','Data Augmentation'] 
        pandas_df_examples = pd.DataFrame(num_examples_obs_categories,columns=cols)
        pandas_df_examples = shuffle(pandas_df_examples)
        pandas_df_examples.reset_index(inplace=True, drop=True)
        print(pandas_df_examples)
        print('\nWriting to {0}\n'.format(training_filename))
        pandas_df_examples.to_csv(training_filename)
        return pandas_df_examples 
    
    def read_files(self,mode,member,dates,hour=None,patch=None,data_augment=None): 
        """
        Function to read pre-processed model data for each individual predictor
        variable.

        Args:
            mode (str): Read training or forecasting files
            member (str): Ensemble member 
            dates (list of strings): 
            hour (str): Random training patch hour
            patch (str): Random training patch
            data_augment (str): Augmentation of random training patch, binary. 
        """
        total_patch_data = []
        
        #Start up parallelization
        client = Client(threads_per_worker=4, n_workers=10) 
        print('\nReading member files\n')
        var_file = self.hf_path + '/{0}/*{1}*{2}*.h5'
        for d,date in enumerate(dates):
            #Find all model variable files for a given date
            var_files = [glob(var_file.format(member,variable,date))[0]
                for variable in self.forecast_variables 
                if len(glob(var_file.format(member,variable,date))) == 1]
            #If at least one variable file is missing, go to next date
            if len(var_files) < len(self.forecast_variables): continue
            if mode == 'train':
                if d%50 == 0:print(d,date)
                #Extract training data 
                total_patch_data.append(dask.delayed(self.extract_data)('train',var_files,hour[d],patch[d],data_augment[d])) 
            #Extract forecast data
            elif mode =='forecast': total_patch_data.append(self.extract_data('forecast',var_files)) 
        
        #If there are no data, return None
        if len(total_patch_data) <1: return None
        elif mode=='train':return np.array(dask.compute(total_patch_data))[0]
        else: return np.array(total_patch_data)[0]

    def extract_data(self,mode,files,hour=None,patch=None,data_augment=None):
        """
        Function to extract training and forecast patch data

        Args:
            mode (str): Training or forecasting
            files (list of str): List of model files
            hour (str): Random training patch hour
            patch (str): Random training patch
            data_augment (str): Augmentation of random training patch, binary. 

        Returns: 
            Extracted data 
        """
        if mode =='train':
            #Training patch data (ny,nx,#variables) 
            patch_data = np.zeros( (self.patch_radius,self.patch_radius,len(self.forecast_variables)) ) 
        else: 
            #Forecast patch data (hours,#patches,ny,nx,#variables) 
            data_shape = h5py.File(files[0],'r')['patches'].shape+(len(self.forecast_variables),)
            patch_data = np.empty( data_shape )*np.nan
        
        for v,variable_file in enumerate(files):
            with h5py.File(variable_file, 'r') as hf:
                if mode =='train':
                    if data_augment > 0.5:
                        #Augment data by adding small amounts of variance
                        var_data = hf['patches'][hour,patch,:,:].ravel()
                        noise = np.nanvar(var_data)*np.random.choice(np.arange(-0.5,0.5,0.15))
                        patch_data[:,:,v] = (var_data + noise).reshape((self.patch_radius,self.patch_radius))
                    else: patch_data[:,:,v] = hf['patches'][hour,patch,:,:]
                else: patch_data[:,:,:,:,v] = hf['patches'][()]
        return patch_data
    
    def standardize_data(self,member,model_data):
        """
        Function to standardize data and output the training 
        mean and standard deviation to apply to testing data.

        Args:
            member (str): Ensemble member 
            model_data (ndarray): Data to standardize
        Returns:
            Standardized data
        """
        
        scaling_file = self.model_path+'/{0}_{1}_{2}_{3}_training_scaling_values.csv'.format(
            member,self.start_dates['train'].strftime('%Y%m%d'),
            self.end_dates['train'].strftime('%Y%m%d'),self.num_examples)

        standard_model_data = np.ones(np.shape(model_data))*np.nan
        #If scaling values are already saved, input data
        if not os.path.exists(scaling_file):
            scaling_values = pd.DataFrame(np.zeros((len(self.forecast_variables), 2), 
                dtype=np.float32),columns=['mean','std'])
        else: scaling_values = pd.read_csv(scaling_file,index_col=0)

        #Standardize data
        for n in np.arange(model_data.shape[-1]):
            if os.path.exists(scaling_file):
                standard_model_data[:,:,:,n] = (model_data[:,:,:,n]-scaling_values.loc[n,'mean'])/scaling_values.loc[n,'std']
                continue
            #Save mean and standard deviation values
            scaling_values.loc[n, ["mean", "std"]] = [model_data[:,:,:,n].mean(),model_data[:,:,:,n].std()]
            standard_model_data[:,:,:,n] = (model_data[:,:,:,n]-scaling_values.loc[n,'mean'])/scaling_values.loc[n,'std']
        
        #Output training scaling values
        if not os.path.exists(scaling_file):
            print('\nWriting to {0}\n'.format(scaling_file))
            scaling_values.to_csv(scaling_file)
        del model_data,scaling_values
        return standard_model_data

    def train_CNN(self,member,model_data,model_labels):
        """
        Function to train a convolutional neural net (CNN) for random 
        training data and associated labels.

        Args:
            member (str): Ensemble member 
            model_data (ndarray): ensemble member data
            model_labels (ndarray): observation labels

        """
        print('\nTraining {0} models'.format(member))
        print('Model data shape {0}'.format(np.shape(model_data)))
        print('Label data shape {0}\n'.format(np.shape(model_labels)))

        #Initiliaze Convolutional Neural Net (CNN)
        model = models.Sequential()
        l2_a= 0.001
        #First layer, input shape (y,x,# variables) 
        model.add(layers.Conv2D(32, (5, 5), activation='relu', 
            kernel_regularizer=l2(l2_a),
            padding='same',input_shape=(np.shape(model_data[0]))))
        model.add(layers.Conv2D(32, (5,5),kernel_regularizer=l2(l2_a),
            activation='relu',padding='same'))
        model.add(layers.Dropout(0.3))
        model.add(layers.MaxPooling2D())
        #Second layer
        model.add(layers.Conv2D(64, (5, 5), 
            kernel_regularizer=l2(l2_a),activation='relu',padding='same'))
        model.add(layers.Conv2D(64, (5,5), 
            kernel_regularizer=l2(l2_a),activation='relu',padding='same'))
        model.add(layers.Dropout(0.3))
        model.add(layers.MaxPooling2D())
        #Third layer
        model.add(layers.Conv2D(128, (5, 5), 
            kernel_regularizer=l2(l2_a),activation='relu',padding='same'))
        model.add(layers.Conv2D(128, (5,5), 
            kernel_regularizer=l2(l2_a),activation='relu',padding='same'))
        model.add(layers.Dropout(0.3))
        model.add(layers.MaxPooling2D())
    
        #Flatten the last convolutional layer into a long feature vector
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(4, activation='softmax'))

        #Compile neural net
        opt = Adam()
        model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['acc'])
        batches = int(self.num_examples/20.0)
        print(batches)
        conv_hist = model.fit(model_data, model_labels, epochs=20, batch_size=batches,validation_split=0.1)
        
        #Save trained model
        model_file = self.model_path+'/{0}_{1}_{2}_CNN_model.h5'.format(member,
            self.start_dates['train'].strftime('%Y%m%d'),self.end_dates['train'].strftime('%Y%m%d'))
        model.save(model_file)
        print('\nWriting out {0}\n'.format(model_file))
        del model_labels,model_data
        return 
    
    def gridded_forecasts(self,forecasts,map_file):
        """
        Function that opens map files over the total data domain
        and the subset domain (CONUS). Projects the patch data on 
        the subset domain to the large domain.
        
        Args:
            forecasts (ndarray): Patch probability predictions from CNN
            map_file (str): Filename and path to total domain map file
        """
        
        #Open mapfile over total domain
        proj_dict, grid_dict = read_ncar_map_file(map_file)
        mapping_data = make_proj_grids(proj_dict, grid_dict)
        #Create cKDtree to convert patch grid points to the total domain
        tree = cKDTree(np.c_[mapping_data['lon'].ravel(),mapping_data['lat'].ravel()])

        #Open mapfile over subset domain (CONUS)
        subset_map_file = glob(self.hf_path+'/*map*.h5')[0] 
        subset_data = h5py.File(subset_map_file, 'r')['map_data']
        #Convert subset grid points to total grid using cKDtree
        _,inds = tree.query(np.c_[subset_data[0].ravel(),subset_data[1].ravel()])
        
        #Fill in total grid with predicted probabilities
        gridded_predictions = np.empty( (forecasts.shape[0],forecasts.shape[1],) +\
            mapping_data['lat'].ravel().shape )*np.nan
        
        #Patch size to fill probability predictions
        subset_data_shape = np.array(subset_data.shape)[-2:]
        
        #Grid to project subset data onto
        total_grid = np.zeros_like( mapping_data['lat'].ravel() )
        
        for size_pred in np.arange(forecasts.shape[0]):
            for hour in np.arange(forecasts.shape[1]):
                prob_on_patches = []
                for patch in np.arange(forecasts.shape[2]):
                    patch_data = forecasts[size_pred,hour,patch]
                    if patch_data < 0.01: patch_data=0.0
                    prob_on_patches.append(np.full(subset_data_shape, patch_data))
                #Output subset data onto full grid using indices from cKDtree
                total_grid[inds] = np.array(prob_on_patches).ravel()
                gridded_predictions[size_pred,hour,:] = total_grid
        
        #Write file out gridded forecasts using Hierarchical Data Format 5 (HDF5) format. 
        final_grid_shape = (forecasts.shape[0],forecasts.shape[1],)+mapping_data['lat'].shape
        return gridded_predictions.reshape(final_grid_shape)

def brier_skill_score_keras(obs, preds):
    bs = k.mean((preds - obs) ** 2)
    climo = k.mean((obs - k.mean(obs)) ** 2)
    return 1.0 - (bs/climo)
