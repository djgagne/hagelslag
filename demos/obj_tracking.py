#!/bin/env python

# coding: utf-8

# Severe Weather Forecasting with Python and Data Science Tools: Interactive Demo
# David John Gagne, University of Oklahoma and NCAR
# Introduction
# Severe weather forecasting has entered an age of unprecedented access to large model and observational datasets with even greater hordes of data in the pipeline. With multiple ensembles of convection-allowing models available and an increasing variety of observations derived from radar, satellite, surface, upper air, and crowd-sourcing, forecasters can easily be overwhelmed with guidance. Without ways to organize, synthesize, and visualize the data in a useful manner for forecasters, the pile of new models and observations will languish unused and will not fulfill their full potential. An even worse outcome would be to take the human forecasters completely out of the loop and trust the models, which is a way fraught with peril. Data science tools offer ways to synthesize essential information from many disparate data sources while also quantifying uncertainty. When forecasters use the tools properly, they can identify potential hazards and the associated spatial and time uncertainties more quickly by using the output of the tools to help target their domain knowledge.
# This module demonstrates how data science tools from the image processing and machine learning families can be used to create a forecast of severe hail. It aims to teach the advantages, challenges, and limitations of these tools through hands-on interaction.
#  

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, PathPatch
from matplotlib.collections import PatchCollection
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter, find_objects
from copy import deepcopy
import pdb, sys, argparse, os


# In[2]:

from hagelslag.processing.EnhancedWatershedSegmenter import EnhancedWatershed
from hagelslag.data import ModelOutput
from hagelslag.processing.ObjectMatcher import ObjectMatcher, closest_distance
from hagelslag.processing import STObject

parser = argparse.ArgumentParser(description='object tracker')
parser.add_argument('-m', '--member', type=str, help='member description (e.g. 1km_pbl7)', default='1km_on_3km_pbl1')
parser.add_argument('-d', '--date', help='date yyyymmddhh', default='2005011312')
parser.add_argument('-f', '--field', default='UP_HELI_MAX03', help='field in which to find objects')
parser.add_argument('-t','--timethresh', type=int, default=3, help='time threshold (hours)')
parser.add_argument('-v','--verbose', action="store_true", help='print more output. useful for debugging')
args = parser.parse_args()

if args.verbose: 
    print(args)

odir = '/glade/p/work/ahijevyc/hagelslag/out/'
model_path = "/glade/scratch/ahijevyc/VSE/"
ensemble_name = "NCAR"
run_date = datetime.strptime(args.date,'%Y%m%d%H')
member = args.member
field = args.field
start_date = run_date + timedelta(hours=10) # remember first time is usually all zeros

#    Attributes:
#        min_thresh (int): minimum pixel value for pixel to be part of a region
#        data_increment (int): quantization interval. Use 1 if you don't want to quantize
#        max_thresh (int): values greater than maxThresh are treated as the maximum threshold
#        size_threshold_pixels (int): clusters smaller than this threshold are ignored.
#        delta (int): maximum number of data increments the cluster is allowed to range over. Larger d results in clusters over larger scales.

# From ahij's config file.
if field == "MAX_UPDRAFT_HELICITY" or field == "UP_HELI_MAX03":
        params = {"min_thresh":75, "step":5, "max_thresh":250, "max_size":50, "delta":75, "min_size":1, "filter_size":0}
if field == "HAIL2D":
        params = {"min_thresh":0.025, "step":0.005, "max_thresh":0.1, "max_size":150, "delta":75, "min_size":0, "filter_size":1}
levels = params['min_thresh'] * np.arange(1,8)
levels = np.append(levels, params['min_thresh'] * 15)
model_watershed_params = (params['min_thresh'],params['step'],params['max_thresh'],params["max_size"],params["delta"])

end_date = start_date + timedelta(hours=0)

from netCDF4 import Dataset
model_grid = ModelOutput(ensemble_name, 
                         member, 
                         run_date, 
                         field, 
                         start_date, 
                         end_date,
                         model_path,
                         single_step=True)
model_map_file="/glade/p/work/ahijevyc/hagelslag/mapfiles/VSE.txt"

model_grid.load_map_info(model_map_file)
model_grid.data = []

d = start_date
deltat = timedelta(minutes=60)

def add_grid(m):
        m.drawstates()
        m.drawcoastlines(linewidth=0.5)
        parallels = np.arange(0.,81.,1.)
        m.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.5)
        meridians = np.arange(0.,351.,2.)
        m.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.5)
        return m
        

while d <= end_date:
        fhr = (d - run_date).total_seconds()/3600
        # list of potential paths to diagnostic files
        dfiles = [model_path+member+run_date.strftime("/%Y%m%d%H")+"/wrf/wrfout_d01_"+d.strftime("%Y-%m-%d_%H:%M:%S"),
                model_path+member+run_date.strftime("/%Y%m%d%H")+"/wrf/diags_d01."+d.strftime("%Y-%m-%d_%H:%M:%S")+".nc",
                model_path+member+run_date.strftime("/%Y%m%d%H")+"/post_AGAIN/"+'fhr_%d'%fhr+"/WRFTWO"+'%02d'%fhr+".nc",
                model_path+member+run_date.strftime("/%Y%m%d%H")+"/wrf/vse_d01."+d.strftime("%Y-%m-%d_%H:%M:%S")+".nc"]
        for dfile in dfiles:
                # see if each path exists
                if not os.path.isfile(dfile):
                        continue
                # If it does, see if 'field' is a variable.
                ncf = Dataset(dfile)
                if field in ncf.variables:
                        print(dfile)
                        model_grid.data.append(ncf.variables[field][0,:,:])
                        ncf.close()
                        break
                ncf.close()
        d += deltat

print(model_grid.lon.shape, np.maximum.reduce(model_grid.data).shape) # max across time dimension 
print(model_grid.data[0].max(), model_grid.data[-1].max(), np.maximum.reduce(model_grid.data).max())

plt.figure(figsize=(10,8))

plt.contourf(model_grid.lon, model_grid.lat,
                np.maximum.reduce(model_grid.data), # max across time dimension 
                levels,
                extend="max",
                latlon= True,
                cmap="Accent")
plt.colorbar(shrink=0.9, fraction=0.1, ticks=levels)
title_info = plt.title(field + "\n"+member+" {0}-{1}".format(start_date.strftime("%d %b %Y %H:%M"),
                        end_date.strftime("%d %b %Y %H:%M")),
                        fontweight="bold", fontsize=14)
dtstr = "_"+member+run_date.strftime("_%Y%m%d%H")
ret = plt.savefig(odir+"uh_swaths/"+field+"_swaths"+dtstr+".png")


def get_forecast_objects(model_grid, ew_params, min_size, gaussian_window):
        ew = EnhancedWatershed(*ew_params)
        model_objects = []
        print("Find model objects Hour:")
        for h in range(int((model_grid.end_date - model_grid.start_date).total_seconds()/deltat.total_seconds())+1):
                print(h)
                hour_labels = ew.size_filter(ew.label(gaussian_filter(model_grid.data[h], gaussian_window)), min_size)
                obj_slices = find_objects(hour_labels)
                num_slices = len(obj_slices)
                model_objects.append([])
                if num_slices > 0:
                        fig, ax = plt.subplots()
                        t = plt.contourf(model_grid.lon,model_grid.lat,hour_labels,np.arange(0,num_slices+1)+0.5,extend="max",cmap="Set1",latlon=True,title=str(run_date)+" "+field+" "+str(h))
                        ret = plt.savefig(odir+"enh_watershed_ex/ew{0:02d}.png".format(h))
                        for s, sl in enumerate(obj_slices): 
                                model_objects[-1].append(STObject(model_grid.data[h][sl],
                                                #np.where(hour_labels[sl] > 0, 1, 0),
                                                # For some objects (especially long, diagonal ones), the rectangular
                                                # slice encompasses part of other objects (i.e. non-zero elements of slice).
                                                # We don't want them in our mask.
                                                np.where(hour_labels[sl] == s+1, 1, 0),
                                                model_grid.x[sl], 
                                                model_grid.y[sl], 
                                                model_grid.i[sl], 
                                                model_grid.j[sl],
                                                h,
                                                h,
                                                dx=model_grid.dx))
                                if h > 0:
                                        dims = model_objects[-1][-1].timesteps[0].shape
                                        model_objects[-1][-1].estimate_motion(h, model_grid.data[h-1], dims[1], dims[0])
        return model_objects

model_objects = get_forecast_objects(model_grid, model_watershed_params, params['min_size'], params['filter_size'])


# In[12]:

def track_forecast_objects(input_model_objects, model_grid, object_matcher):
        model_objects = deepcopy(input_model_objects)
        hours = np.arange(int((model_grid.end_date-model_grid.start_date).total_seconds()/deltat.total_seconds()) + 1)
        print("hours = ",hours)
        tracked_model_objects = []
        for h in hours:
                past_time_objs = []
                for obj in tracked_model_objects:
                        # Potential trackable objects are identified
                        # In other words, objects whose end_time is the previous hour
                        if obj.end_time == h - 1:
                                past_time_objs.append(obj)
                # If no objects existed in the last time step, then consider objects in current time step all new
                if len(past_time_objs) == 0:
                        print("time",h, " no objects existed in the last time step. consider objects in current time step all new")
                        tracked_model_objects.extend(deepcopy(model_objects[h]))
                # Match from previous time step with current time step
                elif len(past_time_objs) > 0 and len(model_objects[h]) > 0:
                        assignments = object_matcher.match_objects(past_time_objs, model_objects[h], h - 1, h)
                        print("assignments:", assignments)
                        unpaired = range(len(model_objects[h]))
                        for pair in assignments:
                                past_time_objs[pair[0]].extend(model_objects[h][pair[1]])
                                unpaired.remove(pair[1])
                        if len(unpaired) > 0:
                                for up in unpaired:
                                    tracked_model_objects.append(model_objects[h][up])
                print("Tracked Model Objects: {0:03d} hour {1:2d}".format(len(tracked_model_objects), h))
        return tracked_model_objects

#object_matcher = ObjectMatcher([shifted_centroid_distance, centroid_distance], 
#                             np.array([dist_weight, 1-dist_weight]), np.array([max_distance] * 2))
object_matcher = ObjectMatcher([closest_distance],np.array([1]),np.array([4*model_grid.dx]))


