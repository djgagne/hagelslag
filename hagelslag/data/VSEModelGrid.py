from .ModelGrid import ModelGrid
from glob import glob
import pandas as pd
from datetime import timedelta
import numpy as np
import os, pdb
from os.path import exists

class VSEModelGrid(ModelGrid):
    """
    Extension of ModelGrid to VSE

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
        self.path = path
        self.member = member
        forecast_hours = np.arange((start_date - run_date).total_seconds() / 3600,
                                   (end_date - run_date).total_seconds() / 3600 + 1)
        # Maybe you need the vse-style file.
        if variable in ["XLAT","XLONG", "MU", "MUB", "Q2", "T2", "PSFC", "U10", "V10", "HGT", "RAINNC", "GRAUPELNC", "HAILNC", "PBLH", "WSPD10MAX", "W_UP_MAX", "W_DN_MAX", "REFD_MAX", "UP_HELI_MAX", "UP_HELI_MAX03", "UP_HELI_MAX01", "UP_HELI_MIN", "RVORT1_MAX", "RVORT0_MAX", "W_MEAN", "GRPL_MAX", "C_PBLH", "W_PBLH", "W_MAX_PBL", "W_1KM_AGL", "HAIL_MAXK1", "HAIL_MAX2D", "PREC_ACC_NC", "REFD_COM", "REFD", "ECHOTOP", "AFWA_MSLP", "AFWA_HAIL", "AFWA_HAIL_NEWMEAN", "AFWA_HAIL_NEWSTD"]:
                full_path = "/".join([self.path, member, run_date.strftime("%Y%m%d%H"), "wrf"])
                potential_filenames = []
                for hour in forecast_hours:
                    valid_time = run_date + timedelta(hours=hour)
                    potential_filenames.append(full_path +"/vse_d01."+valid_time.strftime("%Y-%m-%d_%H:%M:%S")+".nc")

        else: # use the WRFTWO??.nc file
                full_path = "/".join([self.path, member, run_date.strftime("%Y%m%d%H"), "post_AGAIN"])
                potential_filenames = []
                for hour in forecast_hours:
                    potential_filenames.append("{0}/fhr_{1:d}/WRFTWO{2:02d}.nc".format(full_path, int(hour), int(hour) ))
        filenames = []
        for filename in potential_filenames:
            if os.access(filename, os.R_OK):
                filenames.append(filename)
            else:
                print(filename, "not readable. dropping from list")
        super(VSEModelGrid, self).__init__(filenames, run_date, start_date, end_date, variable)
        return
