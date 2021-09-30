import os
from os.path import join

import numpy as np
import pandas as pd

from hagelslag.evaluation import DistributedROC, DistributedReliability

eval_path = "/glade/p/work/dgagne/ncar_coarse_neighbor_eval_2016_s_2/"
eval_files = sorted(os.listdir(eval_path))
eval_test = pd.read_csv(join(eval_path, eval_files[0]))
models = eval_test.columns[eval_test.columns.str.contains("mean")]
run_dates = pd.DatetimeIndex([e.split("_")[-1][:8] for e in eval_files])
thresholds = [25, 50, 75]
prob_thresholds = np.concatenate(([0, 0.01], np.arange(0.1, 1.1, 0.1), [1.05]))
brier = {}
roc = {}
for thresh in thresholds:
    brier[thresh] = pd.DataFrame(index=run_dates, columns=models, dtype=object)
    roc[thresh] = pd.DataFrame(index=run_dates, columns=models, dtype=object)
for ev, eval_file in enumerate(eval_files):
    print(eval_file)
    eval_data = pd.read_csv(join(eval_path, eval_file))
    us_mask = eval_data["us_mask"] == 1
    for thresh in thresholds:
        obs = eval_data.loc[us_mask, "MESH_Max_60min_00.50_{0:2d}".format(thresh)]
        for model in models:
            brier[thresh].loc[run_dates[ev], model] = DistributedReliability(thresholds=prob_thresholds)
            brier[thresh].loc[run_dates[ev], model].update(eval_data.loc[us_mask, model],
                                                           obs)
            roc[thresh].loc[run_dates[ev], model] = DistributedROC(thresholds=prob_thresholds)
            roc[thresh].loc[run_dates[ev], model].update(eval_data.loc[us_mask, model],
                                                         obs)
out_path = "/glade/p/work/dgagne/ncar_coarse_neighbor_scores_2016/"
for thresh in [25, 50, 75]:
    brier[thresh].to_csv(join(out_path, "ncar_2016_s_2_brier_objs_{0:02d}.csv".format(thresh)), index_label="Date")
    roc[thresh].to_csv(join(out_path, "ncar_2016_s_2_roc_objs_{0:02d}.csv".format(thresh)), index_label="Date")
