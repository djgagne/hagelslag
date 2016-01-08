from setuptools import setup

classifiers = ['Development Status :: 4 - Beta',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 2.6',
               'Programming Language :: Python :: 2.7',
              ]


if __name__ == "__main__":
    pkg_description = "Hagelslag is a Python package for storm-based analysis, forecasting, and evaluation."

    setup(name="hagelslag",
          version="0.2",
          description="Object-based severe weather forecast system",
          author="David John Gagne",
          author_email="djgagne@ou.edu",
          long_description=pkg_description,
          license="MIT",
          url="https://github.com/djgagne/hagelslag",
          packages=["hagelslag", "hagelslag.data", "hagelslag.processing", "hagelslag.evaluation", "hagelslag.util"],
          scripts=["bin/hsdata", "bin/hsforecast", "bin/hseval"],
          keywords=["hail", "verification", "tracking", "weather", "meteorology", "machine learning"],
          classifiers=classifiers,
          install_requires=["numpy>=1.9",
                            "pandas>=0.15",
                            "scipy", 
                            "matplotlib", 
                            "netCDF4", 
                            "scikit-learn>=0.16",
                            "basemap",
                            "scikit-image"])
