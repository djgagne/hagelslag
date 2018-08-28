from skimage.draw import polygon
import numpy as np
import shapefile
from netCDF4 import Dataset
from hagelslag.util.make_proj_grids import read_arps_map_file, read_ncar_map_file, make_proj_grids
from pyproj import Proj
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--shape", help="Shape file")
    parser.add_argument("-m", "--map", help="Map projection file")
    parser.add_argument("-o", "--out", help="Output netCDF file")
    args = parser.parse_args()
    print("Loading map info")
    mapping_data, proj_dict, grid_dict = create_map_grid(args.map)
    print("Creating mask grid")
    mask_grid = create_mask_grid(args.shape, mapping_data, proj_dict, grid_dict)
    output_netcdf_file(args.out, mask_grid, proj_dict, grid_dict)
    #plt.figure(figsize=(10, 6))
    #plt.contourf(mapping_data["lon"], mapping_data["lat"], mask_grid, cmap="Reds", vmin=0, vmax=1)
    #plt.show()
    return


def create_map_grid(map_file):
    if map_file[-3:] == "map":
        proj_dict, grid_dict = read_arps_map_file(map_file)
    else:
        proj_dict, grid_dict = read_ncar_map_file(map_file)
    mapping_data = make_proj_grids(proj_dict, grid_dict)
    return mapping_data, proj_dict, grid_dict


def create_mask_grid(mask_shape_file, mapping_data, proj_dict, grid_dict):
    map_proj = Proj(**proj_dict)
    offset_x = mapping_data["x"][0, 0]
    offset_y = mapping_data["y"][0, 0]
    mask_grid = np.zeros(mapping_data["lon"].shape, dtype=int)
    sf = shapefile.Reader(mask_shape_file)
    s = 0
    for state_shape in sf.shapeRecords():
        print(s, state_shape.record[4]) #changed [5])
        part_start = list(state_shape.shape.parts)
        part_end = list(state_shape.shape.parts[1:]) + [len(state_shape.shape.points)]
        for p in range(len(part_start)):
            lon_lat_points = np.array(state_shape.shape.points[part_start[p]:part_end[p]]).T
            sx, sy = map_proj(lon_lat_points[0], lon_lat_points[1])
            si = (sx - offset_x) / grid_dict["dx"]
            sj = (sy - offset_y) / grid_dict["dx"]
            yy, xx = polygon(sj, si)
            valid = np.where((yy < mask_grid.shape[0]) & (xx < mask_grid.shape[1]))
            mask_grid[yy[valid], xx[valid]] = 1
        s += 1
    return mask_grid


def output_netcdf_file(filename, mask_grid, proj_dict, grid_dict):
    out_set = Dataset(filename, "w")
    out_set.createDimension("y", mask_grid.shape[0])
    out_set.createDimension("x", mask_grid.shape[1])
    out_set.set_auto_mask(True)
    var = out_set.createVariable("usa_mask", 'u1', ("y", "x"), zlib=True)
    var[:] = mask_grid
    var.long_name = "USA mask (1 if by land, 0 if by sea)"
    for k, v in proj_dict.items():
        setattr(out_set, k, v)
    for k, v in grid_dict.items():
        setattr(out_set, k, v)
    out_set.close()


if __name__ == "__main__":
    main()
