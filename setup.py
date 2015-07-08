from setuptools import setup

if __name__ == "__main__":
    setup(name="Hagelslag",
          version="0.1",
          description="Object-based severe weather forecast system",
          author="David John Gagne",
          author_email="djgagne@ou.edu",
          packages=["hagelslag", "hagelslag.data", "hagelslag.processing", "hagelslag.evaluation", "hagelslag.util"],
          scripts=["bin/hsdata", "bin/hsforecast", "bin/hseval"],
          install_requires=["numpy>=1.9", 
                            "pandas>=0.15", 
                            "scipy", 
                            "matplotlib", 
                            "netCDF4", 
                            "scikit-learn>=0.16",
                            "basemap"])
