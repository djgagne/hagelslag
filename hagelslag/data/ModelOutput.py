from __future__ import division
from .SSEFModelGrid import SSEFModelGrid
from .VSEModelGrid import VSEModelGrid
from .NCARModelGrid import NCARModelGrid
from .FV3ModelGrid import FV3ModelGrid
from .HRRRModelGrid import HRRRModelGrid
from .HRRREModelGrid import HRRREModelGrid
from .HREFv2ModelGrid import HREFv2ModelGrid
from .NCARStormEventModelGrid import NCARStormEventModelGrid
from hagelslag.util.make_proj_grids import make_proj_grids, read_arps_map_file, read_ncar_map_file, get_proj_obj
from hagelslag.util.derived_vars import relative_humidity_pressure_level, melting_layer_height
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from pyproj import Proj

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
        single_step (bool): If true, each model timestep is in a separate file
            If false, all timesteps are together in the same file.

        map_file (str): path to data map file
    """
    def __init__(self, 
                 ensemble_name, 
                 member_name, 
                 run_date, 
                 variable, 
                 start_date, 
                 end_date,
                 path,
                 map_file,
                 single_step=True):
        self.ensemble_name = ensemble_name
        self.member_name = member_name
        self.run_date = run_date
        self.variable = variable
        self.start_date = start_date
        self.end_date = end_date
        self.start_hour = int((self.start_date - self.run_date).total_seconds()) // 3600
        self.end_hour = int((self.end_date - self.run_date).total_seconds()) // 3600
        self.data = None
        self.valid_dates = None
        self.path = path
        self.map_file = map_file
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
            if self.variable[0:2] == "rh":
                pressure_level = self.variable[2:]
                relh_vars = ["sph", "tmp"]
                relh_vals = {}
                for var in relh_vars:
                    mg = SSEFModelGrid(self.member_name,
                                       self.run_date,
                                       var + pressure_level,
                                       self.start_date,
                                       self.end_date,
                                       self.path,
                                       single_step=self.single_step)
                    relh_vals[var], units = mg.load_data()
                    mg.close()
                self.data = relative_humidity_pressure_level(relh_vals["tmp"],
                                                             relh_vals["sph"],
                                                             float(pressure_level) * 100)
                self.units = "%"
            elif self.variable == "melth":
                input_vars = ["hgtsfc", "hgt700", "hgt500", "tmp700", "tmp500"]
                input_vals = {}
                for var in input_vars:
                    mg = SSEFModelGrid(self.member_name,
                                       self.run_date,
                                       var,
                                       self.start_date,
                                       self.end_date,
                                       self.path,
                                       single_step=self.single_step)
                    input_vals[var], units = mg.load_data()
                    mg.close()
                self.data = melting_layer_height(input_vals["hgtsfc"],
                                                 input_vals["hgt700"],
                                                 input_vals["hgt500"],
                                                 input_vals["tmp700"],
                                                 input_vals["tmp500"])
                self.units = "m"
            else:
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
        elif self.ensemble_name.upper() == "HREFV2":
            mg = HREFv2ModelGrid(self.member_name,
                               self.run_date,
                               self.variable,
                               self.start_date,
                               self.end_date,
                               self.path)
            self.data, self.units = mg.load_data()
        
        elif self.ensemble_name.upper() == "HRRRE":
            mg = HRRREModelGrid(self.member_name,
                               self.run_date,
                               self.variable,
                               self.start_date,
                               self.end_date,
                               self.path,
                               single_step=self.single_step)
            self.data, self.units = mg.load_data()
        
        elif self.ensemble_name.upper() == "SAR-FV3":
            mg = FV3ModelGrid(self.member_name,
                               self.run_date,
                               self.variable,
                               self.start_date,
                               self.end_date,
                               self.path,
                               single_step=self.single_step)
            self.data, self.units = mg.load_data()

        elif self.ensemble_name.upper() == "VSE":
            mg = VSEModelGrid(self.member_name,
                               self.run_date,
                               self.variable,
                               self.start_date,
                               self.end_date,
                               self.path,
                               single_step=self.single_step)
            self.data, self.units = mg.load_data()
            mg.close()
        elif self.ensemble_name.upper() == "HRRR":
            mg = HRRRModelGrid(self.run_date,
                               self.variable,
                               self.start_date,
                               self.end_date,
                               self.path)
            self.data, self.units = mg.load_data()
            mg.close()
        elif self.ensemble_name.upper() == "NCARSTORM":
            mg = NCARStormEventModelGrid(self.run_date,
                                         self.variable,
                                         self.start_date,
                                         self.end_date,
                                         self.path)
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
            for m, v in mapping_data.items():
                setattr(self, m, v)
            self.i, self.j = np.indices(self.lon.shape)
            self.proj = get_proj_obj(proj_dict)
        else:
            proj_dict, grid_dict = read_ncar_map_file(map_file)
            if self.member_name[0:7] == "1km_pbl": # Don't just look at the first 3 characters. You have to differentiate '1km_pbl1' and '1km_on_3km_pbl1'
                grid_dict["dx"] = 1000
                grid_dict["dy"] = 1000
                grid_dict["sw_lon"] = 258.697
                grid_dict["sw_lat"] = 23.999
                grid_dict["ne_lon"] = 282.868269206236
                grid_dict["ne_lat"] = 36.4822338520542 

            self.dx = int(grid_dict["dx"])
            mapping_data = make_proj_grids(proj_dict, grid_dict)
            for m, v in mapping_data.items():
                setattr(self, m, v)
            self.i, self.j = np.indices(self.lon.shape)
            self.proj = get_proj_obj(proj_dict)


    def period_neighborhood_probability(self, radius, smoothing, threshold, stride, x=None, y=None, dx=None):
        """
        Calculate the neighborhood probability over the full period of the forecast

        Args:
            radius: circular radius from each point in km
            smoothing: standard deviation of Gaussian smoother in grid points
            threshold: intensity of exceedance
            stride: number of grid points to skip for reduced neighborhood grid

        Returns:
            neighborhood probabilities
        """
        if x is None:
            x = self.x
            y = self.y
            dx = self.dx
        neighbor_x = x[::stride, ::stride] / 1000.0
        neighbor_y = y[::stride, ::stride] / 1000.0
        neighbor_kd_tree = cKDTree(np.vstack((neighbor_x.ravel(), neighbor_y.ravel())).T)
        neighbor_prob = np.zeros((neighbor_x.shape[0], neighbor_x.shape[1]))
        period_max = self.data.max(axis=0)
        valid_i, valid_j = np.where(period_max >= threshold)
        print(self.variable, len(valid_i))
        if len(valid_i) > 0:
            var_kd_tree = cKDTree(np.vstack((x[valid_i, valid_j] / 1000.0, y[valid_i, valid_j] / 1000.0)).T)
            exceed_points = np.unique(np.concatenate(var_kd_tree.query_ball_tree(neighbor_kd_tree, radius))).astype(int)
            print("Exceed points", len(exceed_points))
            exceed_i, exceed_j = np.unravel_index(exceed_points, neighbor_x.shape)
            neighbor_prob[exceed_i, exceed_j] = 1
            if smoothing > 0:
                neighbor_prob = gaussian_filter(neighbor_prob, smoothing)
        return neighbor_prob

