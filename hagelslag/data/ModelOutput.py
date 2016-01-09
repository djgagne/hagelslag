from SSEFModelGrid import SSEFModelGrid
from NCARModelGrid import NCARModelGrid
from hagelslag.util.make_proj_grids import make_proj_grids, read_arps_map_file, read_ncar_map_file, get_proj_obj
import numpy as np


class ModelOutput(object):
    """
    Container for the model output values and spatial coordinate information.

    Attributes:
        ensemble_name (str): Name of the ensemble being loaded. Currently supports 'NCAR' and 'SSEF'.
        member_name (str): Ensemble member being loaded.
        run_date (datetime): Date of the initial timestep of the model run.
        variable (str): Variable being loaded.
        start_date (datetime.datetime): Date of the first timestep loaded.
        end_date (datetime.datetime): Date of the last timestep loaded.
        path (str): Path to model output
        single_step (bool): If true, each model timestep is in a separate file.
            If false, all timesteps are together in the same file.
    """
    def __init__(self, 
                 ensemble_name, 
                 member_name, 
                 run_date, 
                 variable, 
                 start_date, 
                 end_date,
                 path,
                 single_step=True):
        self.ensemble_name = ensemble_name
        self.member_name = member_name
        self.run_date = run_date
        self.variable = variable
        self.start_date = start_date
        self.end_date = end_date
        self.start_hour = int((self.start_date - self.run_date).total_seconds()) / 3600
        self.end_hour = int((self.end_date - self.run_date).total_seconds()) / 3600
        self.data = None
        self.valid_dates = None
        self.path = path
        self.lat = None
        self.lon = None
        self.x = None
        self.y = None
        self.i = None
        self.j = None
        self.proj = None
        self.dx = None
        self.units = ""
        self.single_step = single_step

    def load_data(self):
        """
        Load the specified variable from the ensemble files, then close the files.
        """
        if self.ensemble_name.upper() == "SSEF":
            mg = SSEFModelGrid(self.member_name,
                               self.run_date,
                               self.variable,
                               self.start_date,
                               self.end_date,
                               self.path,
                               single_step=self.single_step)
            self.data, self.units = mg.load_data()
            mg.close()
        elif self.ensemble_name.upper() == "NCAR":
            mg = NCARModelGrid(self.member_name,
                               self.run_date,
                               self.variable,
                               self.start_date,
                               self.end_date,
                               self.path,
                               single_step=self.single_step)
            self.data, self.units = mg.load_data()
            mg.close()
        else:
            print(self.ensemble_name + " not supported.")

    def load_map_info(self, map_file):
        """
        Load map projection information and create latitude, longitude, x, y, i, and j grids for the projection.

        Args:
            map_file: File specifying the projection information.
        """
        if self.ensemble_name.upper() == "SSEF":
            proj_dict, grid_dict = read_arps_map_file(map_file)
            self.dx = int(grid_dict["dx"])
            mapping_data = make_proj_grids(proj_dict, grid_dict)
            for m, v in mapping_data.iteritems():
                setattr(self, m, v)
            self.i, self.j = np.indices(self.lon.shape)
            self.proj = get_proj_obj(proj_dict)
        elif self.ensemble_name.upper() == "NCAR":
            proj_dict, grid_dict = read_ncar_map_file(map_file)
            self.dx = int(grid_dict["dx"])
            mapping_data = make_proj_grids(proj_dict, grid_dict)
            for m, v in mapping_data.iteritems():
                setattr(self, m, v)
            self.i, self.j = np.indices(self.lon.shape)
            self.proj = get_proj_obj(proj_dict)


