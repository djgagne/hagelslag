.. title:: Getting Started

.. getting_started:

Getting Started
===============

Installation
------------

Hagelslag can be installed directly from ``source``, or it can be downloaded with ``pip``. Hagelslag is compatible with
Python > 3.6.

Source
------
First download hagelslag by cloning the repository from `github <https://github.com/djgagne/hagelslag>`_::
    
    git clone https://github.com/djgagne/hagelslag
    cd hagelslag

Make sure the following dependencies are installed before trying to install hagelslag:

* numpy
* scipy
* s3fs
* matplotlib
* xarray
* netcdf4
* pandas
* scikit-learn
* pytest
* h5py
* pip
* pyproj
* pygrib
* scikit-image
* jupyter
* jupyterlab
* arrow
* cython
* sphinx
* mock
* jasper
* grib2io

Most of these should be installable with the `Anaconda Python distribution <https://www.continuum.io/downloads>`_ or pip.
Pygrib requires the `ECMWF GRIB-API <https://software.ecmwf.int/wiki/display/GRIB/Home>`_.
If you install them all simultaneously, it should handle any potential conflicts.
All hagelslag dependencies can be installed into a conda environment with the following command::

    conda env create -f environment.yml

You can also add depndencies to an existing environment with the following command::

   conda env update -f environment.yml

Quick Start 
------
A stable version of hagelslag can be installed from the Python Package Index (PyPI) with the pip command::

   pip install hagelslag

The latest version from github can be installed with pip as well. After cloning hagelslag with git, run the following command within the hagelslag directory::
    
    pip install .

Example bash scripts for running different aspects of hagelslag are included in ``hagelslag/example_scripts``. 
Any paths that begin with ``...\hagelslag`` should include the directory hagelslag was cloned into.


Before training the desired machine learning models, a model mask file and regridded obervational data onto the model 
domain must be created. The example pre-processing bash script applied to the HREFv2 dataset ::
    
    hagelslag/example_scripts/preprocess_hrefv2_for_ml

requires a map file for the two included python files, examples are found at ``hagelslag/mapfiles``.

Different directories are needed for the tracking and forecast data. Within a desired directory run::
    
    mkdir track_data_2019_MAXUVV_patch_nc track_data_2019_MAXUVV_closest_json track_data_2019_MAXUVV_patch_nc
    mkdir track_forecasts_2019_MAXUVV_closest_json track_forecasts_2019_MAXUVV_closest_csv
    mkdir hail_forecasts_grib2_hrefv2_closest_2019 hail_graphics_hrefv2_MAXUVV_closest_2019 

These directory names are specific to the HREFv2 from 2019, however any name can be used. The only requirement is 
the directories must be included in the configuration files described below. Configuration files within the bash script do not include specific data paths, and will need to be changed to reflect the created directories above and where the mask file, regridded observational data, and model data are stored.

Next, the machine learning models are trained using the example bash script::
    
    hagelslag/example_scripts/train_hrefv2_for_ml

The training script uses ``hsdata`` to pre-process the input training data and ``hsforecast`` with a flag ``-t`` to train the models.

For prediction over a multiple day date range, use::

    hagelslag/example_scripts/multiple_day_forecast_hrefv2

The script includes pre-processing forecast data with ``hsdata``, predicting on the data and regridding the predictions using ``hsforecast`` with the flags ``-f`` and ``-g``. After regridding the predictions, ``hsfileoutput`` outputs the predictions as netcdf files or grib2 files. Calibration is automatically assumed with ``hsfileoutput``. To turn off calibration, include the flag ``-l False``. 

If calibration is desired, include the ``hscalibration`` command within the above bash script. Existing forecast probabilitities are needed to train the calibration method, and therefore cannot be trained and tested on the same datasets as the previous machine learning models in ``hsdata`` and ``hsforecast``. 


If all of the desired machine learning models are trained, including the calibration model, to automatically processing daily model data for calibrated probability predictions, run::

    hagelslag/example_scripts/train_hrefv2_for_ml

Similar to ``multiple_day_forecast_hrefv2``, the configuration files and ``hsfileoutput`` are now evaluated over daily data, given UTC time. 


