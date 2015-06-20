#Hagelslag

Hagelslag is an object-based severe storm forecasting system that utilizing image processing and machine learning tools
to derive calibrated probabilities of severe hazards from convection-allowing numerical weather prediction model output.
The package contains modules for storm identification and tracking, spatio-temporal data extraction, and 
machine learning model training to predict hazard intensity as well as space and time translations.

###Citation
If you employ hagelslag in your research, please acknowledge its use with the following citation:
    
    Gagne, D. J. II, 2015: Severe weather forecasting with python and data science tools. 2015 Unidata Users Workshop,
    Boulder, CO.

If you discover any issues, please post them to the Github issue tracker page. Questions and comments should be sent to
djgagne at ou dot edu.

###Requirements

Hagelslag is easiest to install with the help of the Anaconda Python Distribution, but it should work with other
Python setups as well. Hagelslag requires the following packages and recommends the following versions:

* numpy >= 1.9
* scipy >= 0.15
* matplotlib >= 1.4
* scikit-learn >= 0.16
* pandas >= 0.15
* basemap
* netCDF4-python

###Installation

To install hagelslag, enter the top-level directory of the package and run the standard python setup command: 

    python setup.py install

Hagelslag will install the libraries in site-packages and will also install 3 applications into the `bin` directory
of your Python installation.

###Use
A Jupyter notebook is located in the demos directory that showcases the functionality of the package. For larger scale 
use, 3 scripts are provided in the bin directory. 

* `hsdata` performs object tracking and matching as well as data processing.
* `hsfore` trains and applies machine learning models.
* `hseval` performs forecast verification.

All scripts take input from a config file. The config file should be valid Python code and contain a dictionary called
config. Custom machine learning models and parameters should be contained within the config files. Examples of them can
be found in the config directory.