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
Quick Start will be added here.



