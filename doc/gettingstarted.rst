.. title:: Getting Started

.. getting_started:

Getting Started
===============

Installation
------------

Hagelslag can be installed directly from ``source``, or it can be downloaded with ``pip``. Hagelslag is compatible with
Python 2.7 and 3.5.

Source
------
First download hagelslag by cloning the repository from `github <https://github.com/djgagne/hagelslag>`_::
    
    git clone https://github.com/djgagne/hagelslag
    cd hagelslag

Make sure the following dependencies are installed before trying to install hagelslag:

* numpy
* scipy
* matplotlib
* pandas 
* scikit-learn
* scikit-image
* pyproj
* netCDF4-python
* scikit-image
* pygrib
* basemap

Most of these should be installable with the `Anaconda Python distribution <https://www.continuum.io/downloads>`_ or pip.
Pygrib requires the `ECMWF GRIB-API <https://software.ecmwf.int/wiki/display/GRIB/Home>`_.
Basemap may need to be installed from source because the geos library the anaconda binary uses conflicts with other packages.
Alternatively, use the conda-forge channel in anaconda to install all of the dependencies. If you install them all
simultaneously, it should handle any potential conflicts.

Quick Start 
------
After cloning hagelslag, run the following command within the hagelslag directory::
    python setup.py install

Example bash scripts for running different aspects of hagelslag are included in ``hagelslag/example_scripts``. 
Any paths that begin with ``...\hagelslag`` should include the directory hagelslag was cloned into.


Before training the desired machine learning models, a model mask file and regridded obervational data onto the model 
domain must be created. The example pre-processing bash script that consists of two python files is::
    hagelslag/example_scripts/preprocess_hrefv2_for_ml
A map file is needed for both python files, examples are found at ``hagelslag/mapfiles``.

Different directories are needed for the tracking and forecast data. Within a desired directory run::
    mkdir track_data_2019_MAXUVV_patch_nc track_data_2019_MAXUVV_closest_json/ track_data_2019_MAXUVV_patch_nc/
    mkdir track_forecasts_2019_MAXUVV_closest_json/ track_forecasts_2019_MAXUVV_closest_csv/
    mkdir hail_forecasts_grib2_hrefv2_closest_2019/ hail_graphics_hrefv2_MAXUVV_closest_2019 

These directory names are specific to the HREFv2 from 2019, however any name can be used. The only requirement is 
the directories must be included in the configuration files described below. 

Next, the machine learning models are trained using the example bash script::
    hagelslag/example_scripts/train_hrefv2_for_ml
The configuration files within the bash script do not include specific data paths, and will need to be changed to reflect
the created directories above and where the mask file, regridded observational data, and model data are stored.

