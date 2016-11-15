import pickle
import numpy as np
from hagelslag.util.Config import Config
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config file containing input columns")
    parser.add_argument("-m", "--model", required=True, help="List of machine learning models with feature importances")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of features to display")
    args = parser.parse_args()
    config = Config(args.config)
    input_columns = np.array(config.size_distribution_input_columns)
    model_files = args.model.split(",")
    for model_file in model_files:
        print(model_file)
        model_fo = open(model_file)
        model_obj = pickle.load(model_fo)
        model_fo.close()
        importances = None
        if hasattr(model_obj, "feature_importances_"):
            importances = model_obj.feature_importances_
        elif hasattr(model_obj, "best_estimator_"):
            if hasattr(model_obj.best_estimator_, "feature_importances_"):
                importances = model_obj.best_estimator_.feature_importances_
        if importances is not None:
            ranking = np.argsort(importances)[::-1]
            for r, rank in enumerate(ranking[:args.num]):
                print("{0:2d} {1} {2:1.3f}".format(r, input_columns[rank], importances[rank]))

    return


if __name__ == "__main__":
    main()
