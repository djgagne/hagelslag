import os

from setuptools import setup

classifiers = ['Development Status :: 4 - Beta',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8',
               'Programming Language :: Python :: 3.9',
               ]

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

requires = ["numpy", "matplotlib", "scipy", "xarray", "pandas", "scikit-image", "scikit-learn"]

if __name__ == "__main__":
    pkg_description = "Hagelslag is a Python package for storm-based analysis, forecasting, and evaluation."

    setup(name="hagelslag",
          version="0.5",
          description="Object-based severe weather forecast system",
          author="David John Gagne",
          author_email="dgagne@ucar.edu",
          long_description=pkg_description,
          license="MIT",
          url="https://github.com/djgagne/hagelslag",
          packages=["hagelslag", "hagelslag.data", "hagelslag.processing", "hagelslag.evaluation", "hagelslag.util"],
          scripts=["bin/hsdata", "bin/hsforecast", "bin/hseval", "bin/hsfileoutput", "bin/hsplotter",
                   "bin/hsstation", "bin/hsncarpatch", "bin/hscalibration"],
          data_files=[("mapfiles", ["mapfiles/ssef2013.map",
                                    "mapfiles/ssef2014.map",
                                    "mapfiles/ssef2015.map",
                                    "mapfiles/ncar_grib_table.txt",
                                    "mapfiles/hrrr_map_2016.txt",
                                    "mapfiles/ncar_ensemble_map_2015.txt",
                                    "mapfiles/ncar_2015_us_mask.nc",
                                    "mapfiles/ssef_2015_us_mask.nc",
                                    "mapfiles/ncar_storm_map_3km.txt",
                                    "mapfiles/ncar_storm_us_mask_3km.nc"])],
          keywords=["hail", "verification", "tracking", "weather", "meteorology", "machine learning"],
          classifiers=classifiers,
          include_package_data=True,
          zip_safe=False,
          install_requires=requires)
