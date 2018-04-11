from .ModelGrid import ModelGrid
import numpy as np
from datetime import timedelta


class NCARModelGrid(ModelGrid):
    """
    Extension of the ModelGrid class for interfacing with the NCAR ensemble.

    Args:
        member (str): Name of the ensemble member
        run_date (datetime.datetime object): Date of the initial step of the ensemble run
        start_date (datetime.datetime object): First time step extracted.
        end_date (datetime.datetime object): Last time step extracted.
        path (str): Path to model output files.
        single_step (boolean (default=False)): Whether variable information is stored with each time step in a separate
            file or one file containing all timesteps.
    """
    def __init__(self, member, run_date, variable, start_date, end_date, path, single_step=False):
        self.member = member
        self.path = path
        self.forecast_hours = np.arange((start_date - run_date).total_seconds() / 3600,
                                        (end_date - run_date).total_seconds() / 3600 + 1, dtype=int)
        filenames = []
        if not single_step:
            filenames.append("{0}{1}/{2}_surrogate_{1}.nc".format(self.path,
                                                                  run_date.strftime("%Y%m%d%H"),
                                                                  self.member))
            load_var = variable
        else:
            for hour in self.forecast_hours:
                valid_time = run_date + timedelta(hours=hour)
                if variable in ["UBSHR1", "VBSHR1", "UBSHR6", "VBSHR6", "PWAT", "SRH3", "LCL_HEIGHT", "CAPE_SFC",
                                "CIN_SFC", "MUCAPE"]:
                    filenames.append("{0}{1}/post_rundir/{2}/fhr_{3:d}/WRFTWO{3:02d}.nc".format(self.path,
                                                                                                run_date.strftime(
                                                                                                    "%Y%m%d%H"),
                                                                                                self.member.replace(
                                                                                                    "mem", "mem_"),
                                                                                                int(hour)))
                else:
                    filenames.append("{0}{1}/wrf_rundir/{2}/wrfout_d02_{3}:00:00".format(self.path,
                                                                                         run_date.strftime("%Y%m%d%H"),
                                                                                         self.member.replace("mem",
                                                                                                             "ens_"),
                                                                                         valid_time.strftime(
                                                                                             "%Y-%m-%d_%H")))
            if variable == "SRH3":
                load_var = "SR_HELICITY_3KM"
            elif variable == "CAPE_SFC":
                load_var = "SBCAPE"
            elif variable == "CIN_SFC":
                load_var = "SBCINH"
            else:
                load_var = variable
        super(NCARModelGrid, self).__init__(filenames, run_date, start_date, end_date, load_var)
