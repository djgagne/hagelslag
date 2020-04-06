from multiprocessing import Pool
from datetime import timedelta
from netCDF4 import Dataset
import argparse, pdb
import pandas as pd
import numpy as np
import traceback

import dask
import dask.array as da
from dask.distributed import Client, progress
from itertools import product
import h5py
from glob import glob
import random
import os


class DLPatchSelector(object):
    def __init__(self,model_path,hf_path,
        start_dates,end_dates,num_examples,class_percentages,
        patch_radius,run_date_format,forecast_variables):
        
        self.model_path = model_path
        self.hf_path = hf_path
        self.start_dates = start_dates
        self.end_dates = end_dates
        self.num_examples = num_examples
        self.class_percentages = class_percentages
        self.patch_radius = patch_radius
        self.run_date_format = run_date_format
        self.forecast_variables = forecast_variables
        return


    def load_data(self,member,mode):
        print()
        if mode == 'train':
            training_filename = self.model_path+'/{0}_{1}_{2}_{3}_training_examples.csv'.format(
                member, self.start_dates[mode].strftime('%Y%m%d'),
                self.end_dates[mode].strftime('%Y%m%d'),self.num_examples)
            if not os.path.exists(training_filename):
                category_training_patches = self.selecting_training_examples(member,training_filename)
            else:
                print('\nOpening {0}\n'.format(training_filename))
                category_training_patches = pd.read_csv(training_filename,index_col=0)
            random_date = category_training_patches.loc[:,'Random Date']
            random_hour = category_training_patches.loc[:,'Random Hour']
            random_patch = category_training_patches.loc[:,'Random Patch']
            data_augment = category_training_patches.loc[:,'Data Augmentation']
            
            #Creating label data of shape (# examples, #classes)
            obs_label = category_training_patches.loc[:,'Obs Label']
            member_obs_label = np.zeros((len(obs_label),len(np.unique(obs_label))))
            for l, label in enumerate(obs_label):
                for u,unique in enumerate(np.unique(obs_label)):
                    if label == unique:
                        member_obs_label[l,u] = 1.0 
        
            print('Reading member files:')
            dask_file_open = []
            client = Client(threads_per_worker=4, n_workers=10) 
            for d,date in enumerate(random_date.values):
                if d%500 == 0:
                    print(d,date)
                model_files = [glob(self.hf_path + '/{0}/*{1}*{2}*.h5'.format(member,variable,date))[0] for variable in self.forecast_variables]
                dask_file_open.append(dask.delayed(self.reading_files)(model_files,mode,random_hour[d],random_patch[d],data_augment[d]))
            member_model_data = dask.compute(dask_file_open)[0]
            return np.array(member_model_data), np.array(member_obs_label)
        else:
            string_dates = pd.date_range(start=self.start_dates['forecast'],
                    end=self.end_dates['forecast'],
                    freq='1D').strftime(self.run_date_format)
            member_model_data = np.zeros((len(string_dates),
                        len(self.forecast_variables),
                        self.patch_radius,self.patch_radius))
            for d,date in enumerate(string_dates):
                model_files = [glob(self.hf_path + '/{0}/*{1}*{2}*.h5'.format(member,variable,date))[0] for variable in self.forecast_variables]
                variable_patches = pool.apply_async(reading_files, args=(model_files,mode))
                member_model_data[d] = variable_patches.get()

    def reading_files(self,model_files,mode,hour=None, 
                    patch=None,data_augment=None):
        
        variable_patches = np.zeros((
                                self.patch_radius,self.patch_radius,
                                len(self.forecast_variables)))
        
        for v,variable_file in enumerate(model_files):
            if variable_file:
                with h5py.File(variable_file,'r') as vhf:
                    hf_patch_file = vhf['patches']
                    if mode == 'train':
                        if data_augment == 0:
                            variable_patches[:,:,v] = hf_patch_file[hour,patch,:,:]
                        else:
                            variable_data = hf_patch_file[hour,patch,:,:].flatten()
                            noise = np.nanvar(variable_data)*np.random.choice(np.arange(-0.5,0.5,0.15))
                            variable_patches[:,:,v] = (variable_data + noise).reshape(variable_patches[:,:,v].shape)
                    else:
                        patches = hf_patch_file[()]
            else:continue
        return variable_patches
    
    def selecting_training_examples(self,member,training_filename):
    
        string_dates = pd.date_range(start=self.start_dates['train'],
                    end=self.end_dates['train'],
                    freq='1D').strftime(self.run_date_format)
    
        training_example_catetories = {}
        #Loop through each category:
        for category in [0,1,2,3]:
            all_date_hour_patches_examples = {}
            #Loop through each date
            for str_date in string_dates:
                model_file = glob(self.hf_path + '/{0}/*{1}*'.format(member,str_date))
                #If there are model files with that date, look for obs. Otherwise continue on to next date
                if len(model_file) < 1: continue
                obs_file = glob(self.hf_path + '/*obs*{0}*'.format(str_date))
                if obs_file:
                    #Open obs file
                    with h5py.File(obs_file[0], 'r') as ohf:
                        data = ohf['labels'][()]
                        hourly_data = {}
                        for hour in np.arange(data.shape[0]):
                            if len(np.where(data[hour] == category)[0]) > 1:
                                hourly_data[hour] = np.where(data[hour] == category)[0]
                        if hourly_data:
                            all_date_hour_patches_examples[str_date] = hourly_data
                else: continue
            if all_date_hour_patches_examples:
                training_example_catetories[category] = all_date_hour_patches_examples
        training_example_class_divisions = [] 
        for class_,percentage in self.class_percentages.items():
            number_of_class_examples = int(self.num_examples*percentage)
            if len(training_example_catetories[class_]) < number_of_class_examples:
                data_augment = 1
            else:
                data_augment = 0
            for example in np.arange(number_of_class_examples):
                train_examples = training_example_catetories[class_]
                random_date = np.random.choice(list(train_examples))
                random_hour = np.random.choice(list(train_examples[random_date]))
                random_patch = np.random.choice(list(train_examples[random_date][random_hour]))
                training_example_class_divisions.append([random_date,random_hour,random_patch,class_,data_augment])
        pandas_df_examples = pd.DataFrame(training_example_class_divisions,
            columns=['Random Date','Random Hour', 'Random Patch', 'Obs Label','Data Augmentation']) 
        print(pandas_df_examples)
        print('\nWriting to {0}\n'.format(training_filename))
        pandas_df_examples.to_csv(training_filename)
        return pandas_df_examples 

