import json
from multiprocessing import Pool
from scipy.stats import kstest, gamma
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import os 

def main():
    json_path = "/sharp/djgagne/track_data_spring2015_unique_json/"
    member = "wrf-s3m8_arw"
    run_dates = sorted(os.listdir(json_path))
    ks_frames = []
    pool = Pool(1)
    for run_date in run_dates:
        print(run_date)
        pool.apply_async(run_kstests, (json_path, run_date, member), callback=ks_frames.append)
    pool.close()
    pool.join()
    ks_total_frame = pd.concat(ks_frames)
    ks_total_frame.to_csv("ks_results_2015.csv", index_label="Id")
    return

def run_kstests(json_path, run_date, member):
    try:
        full_path = json_path + "/{0}/{1}/mesh_*.json".format(run_date, member)
        json_files = sorted(glob(full_path))
        ks_results = {"id":[], "ks":[]}
        for json_file in json_files:
            js = open(json_file)
            mesh_track = json.load(js)
            js.close()
            id = mesh_track["properties"]["id"]
            for m, mesh_obj in enumerate(mesh_track["features"]):
                step_id = id + "_{0:03d}".format(m)
                ts = np.array(mesh_obj["properties"]["timesteps"])
                mask = np.array(mesh_obj["properties"]["masks"])
                vals = ts[mask == 1]
                gdist = gamma.fit(vals, floc=vals.min()-0.1)
                sig = kstest(vals, gamma(*gdist).cdf)
                ks_results["id"].append(step_id)
                ks_results["ks"].append(sig)
                if sig[1] < 0.01:
                    print(step_id,)
                    print(sig[1],gdist)
                    print(np.sort(vals))
                    plt.figure(figsize=(8,8))
                    plt.pcolormesh(ts, alpha=0.5, cmap="YlOrRd", vmin=0, vmax=100)
                    pc = plt.pcolormesh(np.ma.array(ts, mask=mask==0), cmap="YlOrRd", vmin=0, vmax=100)
                    plt.title(step_id)
                    plt.colorbar(pc)
                    plt.savefig(step_id + ".png", bbox_inches="tight", dpi=150)
                    plt.close()
        ks_frame = pd.DataFrame(ks_results["ks"], index=ks_results["id"],columns=["D", "p-val"])
        print(ks_frame.shape[0])
    except Exception as e:
        raise e
    return ks_frame
if __name__ == "__main__":
    main()
