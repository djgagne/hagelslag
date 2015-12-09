import cPickle
import numpy as np
from hagelslag.util.Config import Config
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config file containing input columns")
    parser.add_argument("-m", "--model", required=True, help="List of machine learning models with feature importances")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of features to display")
    args = parser.parse_args()
    config = Config(args.config, required_attributes=["size_distribution_input_columns"])
    input_columns = np.array(config.size_distribution_input_columns)
    model_files = args.model.split(",")
    for model_file in model_files:
        print(model_file)
        model_fo = open(model_file)
        model_obj = cPickle.load(model_fo)
        model_fo.close()
        if hasattr(model_obj, "feature_importances_"):
            importances = model_obj.feature_importances_
            ranking = np.argsort(importances)[::-1]
            for r, rank in enumerate(ranking[:args.num]):
                print("{0:2d} {1} {2:1.3f}".format(r, input_columns[rank], importances[rank]))
    return


if __name__ == "__main__":
    main()
