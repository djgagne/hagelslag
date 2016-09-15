
# coding: utf-8

# Severe Weather Forecasting with Python and Data Science Tools: Interactive Demo
# David John Gagne, University of Oklahoma and NCAR
# Introduction
# Severe weather forecasting has entered an age of unprecedented access to large model and observational datasets with even greater hordes of data in the pipeline. With multiple ensembles of convection-allowing models available and an increasing variety of observations derived from radar, satellite, surface, upper air, and crowd-sourcing, forecasters can easily be overwhelmed with guidance. Without ways to organize, synthesize, and visualize the data in a useful manner for forecasters, the pile of new models and observations will languish unused and will not fulfill their full potential. An even worse outcome would be to take the human forecasters completely out of the loop and trust the models, which is a way fraught with peril. Data science tools offer ways to synthesize essential information from many disparate data sources while also quantifying uncertainty. When forecasters use the tools properly, they can identify potential hazards and the associated spatial and time uncertainties more quickly by using the output of the tools to help target their domain knowledge.
# This module demonstrates how data science tools from the image processing and machine learning families can be used to create a forecast of severe hail. It aims to teach the advantages, challenges, and limitations of these tools through hands-on interaction.
#  

# In[8]:

#get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from mpl_toolkits.basemap import Basemap
from scipy.ndimage import gaussian_filter, find_objects
from copy import deepcopy
from glob import glob
from mysavfig import mysavfig
import pdb, sys


# In[2]:

from hagelslag.processing import EnhancedWatershed
from hagelslag.data import ModelOutput
from hagelslag.processing import ObjectMatcher, shifted_centroid_distance, centroid_distance, closest_distance
from hagelslag.processing import STObject


# In[27]:

model_path = "/glade/scratch/ahijevyc/hagelslag/testdata/spring2015_unidata/"
ensemble_name = "SSEF"
member ="wrf-s3cn_arw"
# We will be using the uh_max (hourly max 2-5 km Updraft Helicity) variable for this exercise
# cmpref (simulated composite radar reflectivity) is also available.
variable = "uh_max"


model_path = "/glade/p/mmm/schwartz/VSE/"
model_path = "/glade/scratch/sobash/VSE/"
model_path = "/glade/scratch/ahijevyc/VSE/"
odir = "/glade/p/work/ahijevyc/hagelslag/out/"
ensemble_name = "NCAR"
member ="3km_pbl7"
run_date = datetime(2006, 1, 13, 12)
variable = "MAX_UPDRAFT_HELICITY"
levels = np.arange(20, 120, 10) 
start_date = datetime(2006, 1, 13, 15) # remember first time is usually all zeros
min_thresh = 16

if 1: # remember to change down below too (dfile)
	member = "1km_pbl7"
	run_date = datetime(2005, 12, 28, 12)
	start_date = run_date + timedelta(hours=1) # remember first time is usually all zeros
	min_thresh = 75
	levels = [25,50,75,100,125,150,175,200,250,300,400]
end_date = start_date + timedelta(hours=23)

from netCDF4 import Dataset
model_grid = ModelOutput(ensemble_name, 
			 member, 
			 run_date, 
			 variable, 
			 start_date, 
			 end_date,
			 model_path,
			 single_step=True)


model_grid.data = []

d = start_date
deltat = timedelta(minutes=60)

def add_grid(m):
	m.drawstates()
	m.drawcountries()
	m.drawcoastlines()
	parallels = np.arange(0.,81.,1.)
	m.drawparallels(parallels,labels=[True,False,False,False])
	meridians = np.arange(0.,351.,1.)
	m.drawmeridians(meridians,labels=[False,False,False,True])
	return m
	

while d <= end_date:
	fhr = (d - run_date).total_seconds()/3600
	dfile = model_path+member+run_date.strftime("/%Y%m%d%H")+"/wrf/wrfout_d01_"+d.strftime("%Y-%m-%d_%H:%M:%S")
	dfile = model_path+member+run_date.strftime("/%Y%m%d%H")+"/wrf/diags_d01."+d.strftime("%Y-%m-%d_%H:%M:%S"+".nc")
	dfile = model_path+member+run_date.strftime("/%Y%m%d%H")+"/post_AGAIN/"+'fhr_%d'%fhr+"/WRFTWO"+'%02d'%fhr+".nc"
	print dfile
	ncf = Dataset(dfile)
	if d == start_date:
		model_grid.lon = ncf.variables['longitude'][:]
		model_grid.lat = ncf.variables['latitude'][:]
		model_grid.x = model_grid.lon
		model_grid.y = model_grid.lat
		model_grid.dx = ncf.variables['x'].grid_spacing/1000/100
		model_grid.lat_0 = 30.8 # ncf.getncattr('CEN_LAT')
		model_grid.lon_0 = -90. # ncf.getncattr('CEN_LON')
		model_grid.lat_1 = 30.8 # ncf.getncattr('TRUELAT1')
		model_grid.lat_2 = 30.8 # ncf.getncattr('TRUELAT2')
		model_grid.i, model_grid.j = np.indices(model_grid.lon.shape)
	model_grid.data.append(ncf.variables[variable][0,:,:])
	ncf.close()
	d += deltat

print model_grid.lon.shape, np.maximum.reduce(model_grid.data).shape # max across time dimension 
print model_grid.data[0].max(), model_grid.data[1].max(), np.maximum.reduce(model_grid.data).max()
basemap = Basemap(resolution="l",
				llcrnrlon=model_grid.lon.min()+11.,
				urcrnrlon=model_grid.lon.max()-.2,
				llcrnrlat=model_grid.lat.min()+5.,
				urcrnrlat=model_grid.lat.max()-.5,
		  projection='lcc',lat_1=model_grid.lat_1,lat_2=model_grid.lat_2,
		  lat_0=model_grid.lat_0,lon_0=model_grid.lon_0)
plt.figure(figsize=(10,8))
add_grid(basemap)
basemap.contourf(model_grid.lon, model_grid.lat,
		np.maximum.reduce(model_grid.data), # max across time dimension 
		levels,
		extend="max",
		latlon= True,
		cmap="Accent")
plt.colorbar(shrink=0.9, fraction=0.1, ticks=levels)
title_info = plt.title("Max Updraft Helicity {0}-{1}".format(start_date.strftime("%d %b %Y %H:%M"),
			end_date.strftime("%d %b %Y %H:%M")),
			fontweight="bold", fontsize=14)
dtstr = "_"+member+run_date.strftime("_%Y%m%d%H")
ret = mysavfig(odir+"uh_swaths/uh_swaths"+dtstr+".png")


def ew_demo(min_max, step_val, size_val=50, delta_val=5, time=0):
	ew = EnhancedWatershed(min_max[0],step_val,min_max[1],size_val,delta_val)
	fig = plt.figure()
	add_grid(basemap)
	labels = ew.label(gaussian_filter(model_grid.data[int((time - model_grid.start_date).total_seconds()/deltat.total_seconds())], 1))
	objs = find_objects(labels)
	basemap.contourf(model_grid.lon, model_grid.lat, 
			labels, 
			np.arange(1,labels.max()), 
			cmap="Set1",
			latlon=True)
	for obj in objs:
		sy = model_grid.lat[obj[0].start-1, obj[1].start-1]
		sx = model_grid.lon[obj[0].start-1, obj[1].start-1]
		lon2 = model_grid.lon[obj[0].stop + 1, obj[1].stop + 1]
		lat2 = model_grid.lat[obj[0].stop + 1, obj[1].stop + 1]
		basemap.drawgreatcircle(sx, sy, sx, lat2, del_s=10, color="red")
		basemap.drawgreatcircle(sx, lat2, lon2, lat2, del_s=10, color="red")
		basemap.drawgreatcircle(lon2, lat2, lon2, sy, del_s=10, color="red")
		basemap.drawgreatcircle(lon2, sy, sx, sy, del_s=10, color="red")
	plt.title("Enhanced Watershed ({0:d},{1:d},{2:d},{3:d},{4:d}) Time: {5}".format(min_max[0], 
				                                                       step_val, 
				                                                       min_max[1], 
				                                                       size_val, 
				                                                       delta_val, 
				                                                       time.strftime("%Y-%m-%d %H UTC")))
	axcolor = 'lightgoldenrodyellow'
	ax_size = plt.axes([0.25,0.1,0.65,0.03], axisbg=axcolor)
	size_slider = Slider(ax_size,'size',5,300,valinit=50)
	ret = mysavfig(odir+"enh_watershed_ex/ew_demo{0}.png".format(time.strftime("%m%d%H")))

#ret = ew_demo((5,80),1,size_val=50,delta_val=20,time=model_grid.start_date)


def get_forecast_objects(model_grid, ew_params, min_size, gaussian_window):
	ew = EnhancedWatershed(*ew_params)
	model_objects = []
	#pdb.set_trace()
	print "Find model objects Hour:",
	for h in range(int((model_grid.end_date - model_grid.start_date).total_seconds()/deltat.total_seconds())+1):
		print h,
		hour_labels = ew.size_filter(ew.label(gaussian_filter(model_grid.data[h], gaussian_window)), min_size)
		obj_slices = find_objects(hour_labels)
		num_slices = len(obj_slices)
		model_objects.append([])
		if num_slices > 0:
			fig, ax = plt.subplots()
			add_grid(basemap)
			t = basemap.contourf(model_grid.lon,model_grid.lat,hour_labels,np.arange(0,num_slices+1)+0.5,extend="max",cmap="Set1",latlon=True)
			ret = mysavfig(odir+"enh_watershed_ex/ew{0:02d}.png".format(h))
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

max_thresh = min_thresh*5
step = 10
max_size = 2000
min_size = 6
delta = 60
gaussian_filter_size = 1
model_watershed_params=(min_thresh,step,max_thresh,max_size,delta)
# From Ryan's config file.
gaussian_filter_size = 0
model_watershed_params=(25, 5, 250, 200, 200) #decreasing the size threshold results in more broken tracks

model_objects = get_forecast_objects(model_grid, model_watershed_params, min_size, gaussian_filter_size)


# In[12]:

def track_forecast_objects(input_model_objects, model_grid, object_matcher):
	model_objects = deepcopy(input_model_objects)
	hours = np.arange(int((model_grid.end_date-model_grid.start_date).total_seconds()/deltat.total_seconds()) + 1)
	print "hours = ",hours
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
			print "time",h, " no objects existed in the last time step. consider objects in current time step all new"
			tracked_model_objects.extend(deepcopy(model_objects[h]))
		# Match from previous time step with current time step
		elif len(past_time_objs) > 0 and len(model_objects[h]) > 0:
			assignments = object_matcher.match_objects(past_time_objs, model_objects[h], h - 1, h)
			print "assignments:", assignments
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
#			      np.array([dist_weight, 1-dist_weight]), np.array([max_distance] * 2))
object_matcher = ObjectMatcher([closest_distance],np.array([1]),np.array([5*model_grid.dx]))

tracked_model_objects = track_forecast_objects(model_objects, model_grid, object_matcher)
color_list = ["violet", "cyan", "blue", "green", "purple", "darkgreen", "teal", "royalblue"]
color_arr = np.tile(color_list, len(tracked_model_objects) / len(color_list) + 1)
plt.figure(figsize=(12, 8.5))
add_grid(basemap)
basemap.contourf(model_grid.lon, 
		 model_grid.lat, 
		 np.maximum.reduce(model_grid.data),
		 levels,
		 extend="max",
		 cmap="YlOrRd", latlon=True)
plt.colorbar(shrink=0.8,fraction=0.05)
for t, tracked_model_object in enumerate(tracked_model_objects):
	#pdb.set_trace()
	duration = tracked_model_object.end_time - tracked_model_object.start_time + 1
	if duration <= 2: continue
	# Draw polygon boundaries
	for time in tracked_model_object.times:
		x = tracked_model_object.boundary_polygon(time)[0]
		y = tracked_model_object.boundary_polygon(time)[1]
		basemap.plot(x, y, color=color_arr[t], latlon=True, lw=0.5)
	# Label objects
	traj = tracked_model_object.trajectory()
	xs, ys = basemap(*traj)
	#plt.plot(xs,ys, marker='o', markersize=4, color=color_arr[t], lw=2)
	for lon, lat, x, y, time, u, v in zip(traj[0], traj[1], xs,ys,tracked_model_object.times,tracked_model_object.u,tracked_model_object.v):
		print "#",t," lon,lat=",lon,lat,"time=",time,"u,v=",u,v
		#plt.text(x,y,str(time)+":"+str(t), fontsize=7)
		plt.text(x,y,str(time), fontsize=7)
		#plt.barbs(x,y,u/model_grid.dx, v/model_grid.dx, length=6, barbcolor=color_arr[t])
plt.title(title_info.get_text(), fontweight="bold", fontsize=14)
ret = mysavfig(odir+"storm_tracks/storm_tracks"+dtstr+".png")

