from SSEFModelGrid import SSEFModelGrid
from NCARModelGrid import NCARModelGrid
from hagelslag.util.make_proj_grids import make_proj_grids, read_arps_map_file, read_ncar_map_file, get_proj_obj
import numpy as np


class ModelOutput(object):
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
        self.start_hour = (self.start_date - self.run_date).total_seconds() / 3600
        self.end_hour = (self.end_date - self.run_date).total_seconds() / 3600
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
        self.single_step = single_step

    def load_data(self):
        if self.ensemble_name.upper() == "SSEF":
            mg = SSEFModelGrid(self.member_name,
                               self.run_date,
                               self.variable,
                               self.start_date,
                               self.end_date,
                               self.path,
                               single_step=self.single_step)
            self.data = mg.load_data()
            mg.close()
        elif self.ensemble_name.upper() == "NCAR":
            mg = NCARModelGrid(self.member_name,
                               self.run_date,
                               self.variable,
                               self.start_date,
                               self.end_date,
                               self.path,
                               single_step=self.single_step)
            self.data = mg.load_data()
            mg.close()

    def load_map_info(self, map_file):
        if self.ensemble_name.upper() == "SSEF":
            proj_dict, grid_dict = read_arps_map_file(map_file)
            mapping_data = make_proj_grids(proj_dict, grid_dict)
            for m, v in mapping_data.iteritems():
                setattr(self, m, v)
            self.i, self.j = np.indices(self.lon.shape)
            self.proj = get_proj_obj(proj_dict)
        elif self.ensemble_name.upper() == "NCAR":
            proj_dict, grid_dict = read_ncar_map_file(map_file)
            mapping_data = make_proj_grids(proj_dict, grid_dict)
            for m, v in mapping_data.iteritems():
                setattr(self, m, v)
            self.i, self.j = np.indices(self.lon.shape)
            self.proj = get_proj_obj(proj_dict)






