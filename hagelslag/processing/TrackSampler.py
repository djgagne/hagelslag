from netCDF4 import Dataset
import numpy as np
import pandas as pd
import json
from glob import glob
import os
from datetime import datetime
from multiprocessing import Pool
import argparse
import pickle
from scipy.ndimage import binary_dilation
from scipy.stats import multivariate_normal
import traceback


def load_grid_info(grid_file):
    grid_info = {}
    grid_obj = Dataset(grid_file)
    for var in grid_obj.variables.keys():
        grid_info[var] = grid_obj.variables[var][:]
    grid_obj.close()
    return grid_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dates", help="Model Run dates YYYYMMDD")
    parser.add_argument("-p", "--proc", type=int, help="Number of processors")
    args = parser.parse_args()
    run_dates_str = args.run_dates.split(",")
    run_dates = [datetime.strptime(rd, "%Y%m%d") for rd in run_dates_str]
    members = ["wrf-s3cn_arw"] + ["wrf-s3m{0:d}_arw".format(m) for m in range(3, 14)]
    model_names = ["Random-Forest", "Gradient-Boosting", "Random-Forest-Weighted", "Random-Forest-Big"]
    start_hour = 12
    end_hour = 36
    dx = 3000
    num_samples = 1000
    thresholds = np.array([0, 25, 50])
    track_path = "/sharp/djgagne/track_forecasts_spring2015_json/"
    out_path = "/sharp/djgagne/track_samples_spring2015/"
    grid_filename = "/sharp/djgagne/ssef_2015_grid.nc"
    copula_file = "/sharp/djgagne/track_copulas.pkl"
    member_info_file = "/sharp/djgagne/member_info_ssef_spring2015.csv"
    member_info = pd.read_csv(member_info_file, index_col="Ensemble_Member")
    grid_info = load_grid_info(grid_filename)
    grid_shape = grid_info["lon"].shape
    if args.proc > 1:
        pool = Pool(args.proc)
        try:
            for run_date in run_dates:
                for member in members:
                    group = member_info.loc[member, "Microphysics"]
                    pool.apply_async(sample_member_run_tracks, (member,
                                                                group,
                                                                run_date,
                                                                model_names,
                                                                start_hour,
                                                                end_hour,
                                                                grid_shape,
                                                                dx,
                                                                track_path,
                                                                num_samples,
                                                                thresholds,
                                                                copula_file,
                                                                out_path
                                                                ))
            pool.close()
            pool.join()
        finally:
            pool.terminate()
            pool.join()
    else:
        for run_date in run_dates:
            for member in members:
                group = member_info.loc[member, "Microphysics"]
                apply(sample_member_run_tracks, (member,
                                                 group,
                                                 run_date,
                                                 model_names,
                                                 start_hour,
                                                 end_hour,
                                                 grid_shape,
                                                 dx,
                                                 track_path,
                                                 num_samples,
                                                 thresholds,
                                                 copula_file,
                                                 out_path
                                                 ))

    return


def sample_member_run_tracks(member,
                             group,
                             run_date,
                             model_names,
                             start_hour,
                             end_hour,
                             grid_shape,
                             dx,
                             track_path,
                             num_samples,
                             thresholds,
                             copula_file,
                             out_path,
                             size_ranges,
                             track_ranges):
    try:
        ts = TrackSampler(member,
                          group,
                          run_date,
                          model_names,
                          start_hour,
                          end_hour,
                          grid_shape,
                          dx,
                          track_path,
                          num_samples,
                          copula_file=copula_file
                          )
        ts.load_track_forecasts()
        track_probs = ts.sample_tracks(size_ranges, track_ranges, thresholds)
        ts.output_track_probs(track_probs, out_path)
    except Exception as e:
        print(traceback.format_exc())
        raise e
    return


class TrackSampler(object):
    """
    Monte Carlo sampler of forecast storm tracks.

    """
    def __init__(self, member, group, run_date, model_names, start_hour,
                 end_hour, grid_shape, dx, track_path, num_samples, copula_file=None):
        self.member = member
        self.group = group
        self.run_date = run_date
        self.model_names = [mn.replace(" ", "-") for mn in model_names]
        self.track_path = track_path
        self.num_samples = num_samples
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.grid_shape = grid_shape
        self.dx = int(dx)
        self.copula_file = copula_file
        if self.copula_file is not None:
            with open(self.copula_file) as copula_obj:
                self.copula = pickle.load(copula_obj)
        self.track_forecasts = []

    def load_track_forecasts(self):
        run_date_str = self.run_date.strftime("%Y%m%d")
        track_files = sorted(glob(self.track_path + "/".join([run_date_str, self.member]) + "/*.json"))
        for track_file in track_files:
            tfo = open(track_file)
            self.track_forecasts.append(json.load(tfo))
            tfo.close()

    def generate_copula_ranks(self):
        copula_samples = multivariate_normal.rvs(self.copula[self.group]['mean'],
                                                 self.copula[self.group]['cov'],
                                                 self.num_samples)
        ranks = np.argsort(copula_samples, axis=0)
        return pd.DataFrame(data=ranks, columns=self.copula[self.group]['model_names'])

    def sample_tracks(self, size_ranges, track_ranges, thresholds=np.array([0, 25, 50]), dilation=13):
        track_probs = {}
        print(self.member, "Sample condition")
        condition_samples = self.sample_condition()
        print(self.member, "Sample size")
        size_values = np.arange(size_ranges[0], size_ranges[1] + size_ranges[2], size_ranges[2], dtype=int)
        size_samples = self.sample_size(size_values=size_values)
        print(self.member, "Sample start time")
        start_time_values = np.arange(track_ranges['start-time'][0],
                                      track_ranges['start-time'][1] + track_ranges['start-time'][2],
                                      track_ranges['start-time'][2], dtype=int)
        st_samples = self.sample_start_time(start_time_values=start_time_values)
        print(self.member, "Sample translation")
        translation_x_values = np.arange(track_ranges['translation-x'][0],
                                         track_ranges['translation-x'][1] + track_ranges['translation-x'][2],
                                         track_ranges['translation-x'][2], dtype=int)
        translation_y_values = np.arange(track_ranges['translation-y'][0],
                                         track_ranges['translation-y'][1] + track_ranges['translation-y'][2],
                                         track_ranges['translation-y'][2], dtype=int)
        tx_samples = self.sample_translation_x(translation_x_values=translation_x_values)
        ty_samples = self.sample_translation_y(translation_y_values=translation_y_values)
        if self.copula_file is not None:
            copula_ranks = self.generate_copula_ranks()
        for model_name in self.model_names:
            print(model_name)
            track_probs[model_name] = {}
            for thresh in thresholds:
                track_probs[model_name][thresh] = np.zeros((self.end_hour - self.start_hour + 1,
                                                            self.grid_shape[0],
                                                            self.grid_shape[1]))
            for t, track in enumerate(self.track_forecasts):
                print(model_name, self.member, self.run_date, t)
                start_time = track['properties']['times'][0]
                st_track_sample = np.zeros(st_samples[model_name][t].shape, dtype=int)
                tx_track_sample = np.zeros(tx_samples[model_name][t].shape, dtype=int)
                ty_track_sample = np.zeros(ty_samples[model_name][t].shape, dtype=int)
                if self.copula_file is not None:
                    print("copula being used")
                    st_track_sample[copula_ranks['start-time'].values] = np.sort(st_samples[model_name][t])
                    tx_track_sample[copula_ranks['translation-x'].values] = np.sort(tx_samples[model_name][t])
                    ty_track_sample[copula_ranks['translation-y'].values] = np.sort(ty_samples[model_name][t])
                else:
                    print("copula not being used")
                    st_track_sample = st_samples[model_name][t]
                    tx_track_sample = st_samples[model_name][t]
                    ty_track_sample = st_samples[model_name][t]
                for f, feature in enumerate(track['features']):
                    i_track = np.array(feature['properties']['i'], dtype=int)
                    j_track = np.array(feature['properties']['j'], dtype=int)
                    j_track_big, i_track_big = np.meshgrid(
                        np.arange(j_track.min() - dilation, j_track.max() + dilation + 1),
                        np.arange(i_track.min() - dilation, i_track.max() + dilation + 1))
                    mask = np.array(feature['properties']['masks'], dtype=int)
                    mask_big = np.zeros((mask.shape[0] + dilation * 2, mask.shape[1] + dilation * 2))
                    mask_big[dilation: (dilation + mask.shape[0]), dilation: (dilation + mask.shape[1])] = mask
                    mask_big[binary_dilation(mask_big, iterations=dilation)] = 1
                    i_good = i_track_big[mask_big > 0]
                    j_good = j_track_big[mask_big > 0]
                    for s in range(self.num_samples):
                        st_corr = start_time + st_track_sample[s]
                        if (condition_samples[model_name][t][f][s] > 0) and (st_corr >= self.start_hour) \
                                and (st_corr <= self.end_hour):
                            tj_corr = tx_track_sample[s] // self.dx
                            ti_corr = ty_track_sample[s] // self.dx
                            size_sample = size_samples[model_name][t][f][s]
                            i_adj = i_good + ti_corr
                            j_adj = j_good + tj_corr
                            ij_valid = (i_adj < self.grid_shape[0]) & (i_adj >= 0) & \
                                       (j_adj < self.grid_shape[1]) & (j_adj >= 0)
                            i_adj = i_adj[ij_valid]
                            j_adj = j_adj[ij_valid]
                            for thresh in thresholds:
                                if size_sample >= thresh and ((st_corr + f) <= self.end_hour):
                                    track_probs[model_name][thresh][st_corr + f - self.start_hour,
                                                                    i_adj, j_adj] += 1
            for thresh in thresholds:
                track_probs[model_name][thresh] /= float(self.num_samples)
        return track_probs

    def sample_condition(self):
        condition_samples = {}
        for model_name in self.model_names:
            condition_samples[model_name] = []
            for t, track in enumerate(self.track_forecasts):
                condition_samples[model_name].append([])
                for f, feature in enumerate(track['features']):
                    cond_prob = feature['properties']['condition_' + model_name]
                    condition_samples[model_name][t].append(np.random.choice([0, 1],
                                                                             self.num_samples,
                                                                             p=[1 - cond_prob, cond_prob]))
        return condition_samples

    def sample_size(self, size_values=np.arange(5, 105, 5)):
        size_samples = {}
        for model_name in self.model_names:
            size_samples[model_name] = []
            for t, track in enumerate(self.track_forecasts):
                size_samples[model_name].append([])
                for f, feature in enumerate(track['features']):
                    size_prob = feature['properties']['size_' + model_name]
                    size_samples[model_name][t].append(np.random.choice(size_values,
                                                                        self.num_samples,
                                                                        p=size_prob))
        return size_samples

    def sample_start_time(self, start_time_values=np.arange(-6, 7, 1)):
        st_samples = {}
        for model_name in self.model_names:
            st_samples[model_name] = []
            for t, track in enumerate(self.track_forecasts):
                st_prob = track['features'][0]['properties']['start-time_' + model_name]
                st_samples[model_name].append(np.random.choice(start_time_values,
                                                               self.num_samples,
                                                               p=st_prob))
        return st_samples

    def sample_translation_x(self, translation_x_values=np.arange(-192000, 192000 + 24000, 24000)):
        tx_samples = {}
        for model_name in self.model_names:
            tx_samples[model_name] = []
            for t, track in enumerate(self.track_forecasts):
                tx_prob = track['features'][0]['properties']['translation-x_' + model_name]
                tx_samples[model_name].append(np.random.choice(translation_x_values,
                                                               self.num_samples,
                                                               p=tx_prob) +
                                              np.random.randint(0,
                                                                translation_x_values[1] - translation_x_values[0],
                                                                self.num_samples))
        return tx_samples

    def sample_translation_y(self, translation_y_values=np.arange(-192000, 192000 + 24000, 24000)):
        ty_samples = {}
        for model_name in self.model_names:
            ty_samples[model_name] = []
            for t, track in enumerate(self.track_forecasts):
                ty_prob = track['features'][0]['properties']['translation-y_' + model_name]
                ty_samples[model_name].append(np.random.choice(translation_y_values,
                                                               self.num_samples,
                                                               p=ty_prob) +
                                              np.random.randint(0,
                                                                translation_y_values[1] - translation_y_values[0],
                                                                self.num_samples))
        return ty_samples

    def output_track_probs(self, track_probs, path):
        run_date_str = self.run_date.strftime("%Y%m%d")
        if not os.access(path + run_date_str, os.R_OK):
            try:
                os.mkdir(path + run_date_str)
            except OSError:
                print(path + run_date_str + " already created")
        for model in self.model_names:
            filename = path + "{2}/{0}_hailprobs_{1}_{2}.nc".format(model, self.member, run_date_str)
            out_obj = Dataset(filename, "w")
            out_obj.createDimension("hour", self.end_hour - self.start_hour + 1)
            out_obj.createDimension("y", self.grid_shape[0])
            out_obj.createDimension("x", self.grid_shape[1])
            forecast_hour = out_obj.createVariable("forecast_hour", "i4", ("hour",))
            forecast_hour[:] = np.arange(self.start_hour, self.end_hour + 1)
            forecast_hour.units = "hour"
            for thresh in track_probs[model].keys():
                var_obj = out_obj.createVariable("prob_hail_{0:02d}_mm".format(thresh),
                                                 "f4",
                                                 ("hour", "y", "x"), zlib=True)
                var_obj[:] = track_probs[model][thresh]
                var_obj.units = ""
                var_obj.long_name = "probability of hail at least {0:02d} mm diameter".format(thresh)
            out_obj.model_name = model
            out_obj.run_date = run_date_str
            out_obj.close()
        return


if __name__ == "__main__":
    main()
