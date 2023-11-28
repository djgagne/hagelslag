from .ModelGrid import ModelGrid
import numpy as np
from os.path import join
class MPASHrrrGrid(ModelGrid):
    """
    Loads model output from the NCAR MMM 1 and 3 km WRF runs on Cheyenne.

    """

    def __init__(self, run_date, variable, start_date, end_date, member, path):
        self.path = path
        self.forecast_hours = np.arange((start_date - run_date).total_seconds() / 3600,
                                        (end_date - run_date).total_seconds() / 3600 + 1, dtype=int)
        filenames = []
        for hour in self.forecast_hours:
            filename = join(path, run_date.strftime("%Y%m%d%H"), "post", f"mem_{member}",
                            f"interp_mpas_3km_{run_date.strftime('%Y%m%d%H')}_mem{member}_f{hour:03}.nc")
            filenames.append(filename)
        super(MPASHrrrGrid, self).__init__(filenames, run_date, start_date, end_date, variable)

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