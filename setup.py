from setuptools import setup

if __name__ == "__main__":
    setup(name="Hagelslag",
          version="0.1",
          description="Object-based severe weather forecast system",
          author="David John Gagne",
          author_email="djgagne@ou.edu",
          packages=["hagelslag"],
          install_requires=["numpy>=1.9", 
                            "pandas>=0.15", 
                            "scipy", 
                            "matplotlib", 
                            "netCDF4", 
                            "scikit-learn>=0.16"])
