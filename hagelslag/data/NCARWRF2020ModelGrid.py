import numpy as np
from .ModelGrid import ModelGrid
from datetime import timedelta
from os.path import join


class NCARWRF2020ModelGrid(ModelGrid):
    """
    Loads model output from the NCAR MMM WRF 2020 3 km real-time runs

    """
    def __init__(self, member_name, run_date, variable, start_date, end_date, path):
        self.member_name = member_name
        self.pressure_levels = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100])
        self.path = path
        self.forecast_hours = np.arange((start_date - run_date).total_seconds() // 3600,
                                        (end_date - run_date).total_seconds() // 3600 + 1, dtype=int)
        filenames = []
        for hour in self.forecast_hours:
            filename = join(path, run_date.strftime("%Y%m%d%H"), member_name,
                            "diags_d01_f{0:03d}.nc".format(hour))
            filenames.append(filename)
        super(NCARWRF2020ModelGrid, self).__init__(filenames, run_date, start_date, end_date, variable)

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
