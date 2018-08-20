from netCDF4 import Dataset

data = Dataset('NP_2018_mask.nc')

print(data.variables["usa_mask"])
