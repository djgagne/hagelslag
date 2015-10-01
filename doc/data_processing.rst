.. title:: Data Processing

.. data_processing:

Data Processing
===============
The Hagelslag library can read gridded model and observation data with the purpose of identifying storm objects and tracks. Once these tracks are identified, data can be extracted from within the bounds of the tracks.

Model Output
------------
Hagelslag supports reading model output in netCDF format with the netCDF4-python library. NetCDF files from the CAPS Storm-Scale Ensemble and the NCAR Ensemble are currently supported. Other model output could be supported by creating a file handler that extends the ModelGrid class.   

Object Finding
--------------

Object Tracking
---------------
