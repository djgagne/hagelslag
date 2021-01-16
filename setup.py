from setuptools import setup
import os

classifiers = ['Development Status :: 4 - Beta',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               ]

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    requires = []
else:
    with open("requirements.txt") as require_file:
        requires = [r.strip() for r in require_file.readlines()]

if __name__ == "__main__":
    pkg_description = "Hagelslag is a Python package for storm-based analysis, forecasting, and evaluation."

    setup(name="hagelslag",
          version="0.4.0b1",
          description="Object-based severe weather forecast system",
          author="David John Gagne",
          author_email="dgagne@ucar.edu",
          long_description=pkg_description,
          license="MIT",
          url="https://github.com/djgagne/hagelslag",
          packages=["hagelslag", "hagelslag.data", "hagelslag.processing", "hagelslag.evaluation", "hagelslag.util"],
          scripts=["bin/hsdata", "bin/hsforecast", "bin/hseval", "bin/hsfileoutput", "bin/hsplotter", 
                "bin/hswrf3d", "bin/hsstation", "bin/hsncarpatch", "bin/hscalibration"],
          data_files=[("mapfiles", ["mapfiles/ssef2013.map", 
                                    "mapfiles/ssef2014.map", 
                                    "mapfiles/ssef2015.map", 
                                    "mapfiles/ncar_grib_table.txt",
                                    "mapfiles/hrrr_map_2016.txt",
                                    "mapfiles/ncar_ensemble_map_2015.txt",
                                    "mapfiles/ncar_2015_us_mask.nc",
                                    "mapfiles/ssef_2015_us_mask.nc"])],
          keywords=["hail", "verification", "tracking", "weather", "meteorology", "machine learning"],
          classifiers=classifiers,
          include_package_data=True,
          zip_safe=False,
          install_requires=requires)
