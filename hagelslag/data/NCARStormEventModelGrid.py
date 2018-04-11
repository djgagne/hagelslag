import numpy as np
from .ModelGrid import ModelGrid
from datetime import timedelta
from os.path import join


class NCARStormEventModelGrid(ModelGrid):
    """
    Loads model output from the NCAR MMM 1 and 3 km WRF runs on Cheyenne.

    """
    def __init__(self, run_date, variable, start_date, end_date, path):
        self.pressure_levels = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100])
        self.path = path
        self.forecast_hours = np.arange((start_date - run_date).total_seconds() / 3600,
                                        (end_date - run_date).total_seconds() / 3600 + 1, dtype=int)
        filenames = []
        for hour in self.forecast_hours:
            valid_time = run_date + timedelta(hours=int(hour))
            filename = join(path, run_date.strftime("%Y%m%d%H"),
                            "diags_d01_{0}.nc".format(valid_time.strftime("%Y-%m-%d_%H_%M_%S")))
            filenames.append(filename)
        super(NCARStormEventModelGrid, self).__init__(filenames, run_date, start_date, end_date, variable)

    def format_var_name(self, variable, var_list):
        z_index = None
        if variable in var_list:
            var_name = variable
        elif "_PL" in variable:
            var_parts = variable.split("_")
            var_name = "_".join(var_parts[:-1])
            p_level = int(var_parts[-1])
            z_index = np.where(self.pressure_levels == p_level)[0][0]
        else:
            raise KeyError("{0} not found in {1}".format(variable, ", ".join(var_list)))
        return var_name, z_index
