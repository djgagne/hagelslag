import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from glob import glob
import argparse
from os.path import join


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to storm patch files")
    parser.add_argument("-n", "--nprocs", type=int, help="Number of processors")
    parser.add_argument("-o", "--out", help="Output path")
    args = parser.parse_args()
    patch_files = sorted(glob(join(args.path, "*.nc")))
    patch_center_list = [get_storm_centers(patch_file) for patch_file in patch_files]
    patch_center_frame = pd.concat(patch_center_list, ignore_index=True)
    patch_center_frame.to_csv(args.out, index=False)

def get_storm_centers(patch_file):
    patch_info = patch_file.split("/")[-1][:-3].split("_")
    run_date = pd.Timestamp(patch_info[-3][:-2])
    member = patch_info[-1]
    patch_data = None
    print(run_date, member)
    with Dataset(patch_file) as patch_set:
       num_patches = patch_set.variables["p"].size
       lons = patch_set.variables["longitude"][:, 32, 32]
       lats = patch_set.variables["latitude"][:, 32, 32]
       valid_date = num2date(patch_set.variables["valid_date"][:], patch_set.variables["valid_date"].units)
       patch_data = pd.DataFrame(dict(longitude=lons, 
                                      latitude=lats, 
                                      member=[member] * num_patches, 
                                      run_date=[run_date] * num_patches,
                                      valid_date=valid_date), columns=["run_date", "member", "valid_date", "longitude", "latitude"])
    return patch_data

if __name__ == "__main__":
    main()
