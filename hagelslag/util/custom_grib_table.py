import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input csv file")
    parser.add_argument("-o", "--out", help="Output grib table file")
    args = parser.parse_args()
    grib_csv_table = pd.read_csv(args.input, skiprows=5)
    grib_csv_table.loc[:, "mtab_set"] = 1
    grib_csv_table.loc[:, "mtab_low"] = 0
    grib_csv_table.loc[:, "mtab_high"] = 10
    grib_csv_table.loc[:, "CenterNumber"] = 59
    grib_csv_table.loc[:, "local_version"] = 0
    # Order info from http://www.cpc.ncep.noaa.gov/products/wesley/wgrib2/user_grib2tables.html
    out_order = ["DisciplineNumber", "mtab_set", "mtab_low", "mtab_high", "CenterNumber", "local_version",
                 "CategoryNumber", "ParameterNumber", "NCLName", "Description", "Units"]
    grib_out_table = grib_csv_table[out_order]
    print(grib_out_table)
    grib_out_table.to_csv(args.out, sep=":", header=None, index=False)
    return


if __name__ == "__main__":
    main()