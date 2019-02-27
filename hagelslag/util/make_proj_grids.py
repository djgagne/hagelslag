from pyproj import Proj
import numpy as np


def main():
    map_filename = '../ssef2015.map'
    proj_dict, grid_dict = read_arps_map_file(map_filename)
    print(proj_dict)
    print(grid_dict)
    print("Proj")
    mapping_data = make_proj_grids(proj_dict, grid_dict)
    print(mapping_data['lon'].shape)
    print(mapping_data['x'].shape)


def read_arps_map_file(map_filename):
    with open(map_filename) as map_file:
        map_params = map_file.read().split()
    proj_dict = {'proj': 'lcc', 'a': 6370000.0, 'b': 6370000.0, 'units': 'm'}
    grid_dict = {}
    proj_names = ['lat_2', 'lat_1', 'lat_0', 'lon_0']
    grid_names = ['sw_lat', 'sw_lon', 'ne_lat', 'ne_lon', 'dx', 'dy']
    for i in range(len(proj_names) + len(grid_names)):
        if i < len(proj_names):
            proj_dict[proj_names[i]] = float(map_params[i+2])
        else:
            j = i - len(proj_names)
            grid_dict[grid_names[j]] = float(map_params[i+2])
    return proj_dict, grid_dict


def read_ncar_map_file(map_filename):
    proj_keys = ["proj", "a", "b", "lat_2", "lat_1", "lat_0", "lon_0", "units"]
    grid_keys = ["sw_lat", "sw_lon", "ne_lat", "ne_lon", "dx", "dy"]
    proj_dict = {}
    grid_dict = {}
    with open(map_filename) as map_file:
        for line in map_file:
            map_option = line.split("=")
            if map_option[0] in ["a", "b", "lat_2", "lat_1", "lat_0", "lon_0",
                                 "sw_lat", "sw_lon", "ne_lat", "ne_lon", "dx", "dy"]:
                map_option[1] = float(map_option[1].strip())
            else:
                map_option[1] = map_option[1].strip()
            if map_option[0] in proj_keys:
                proj_dict[map_option[0]] = map_option[1]
            elif map_option[0] in grid_keys:
                grid_dict[map_option[0]] = map_option[1]
    return proj_dict, grid_dict


def make_proj_grids(proj_dict, grid_dict):
    map_proj = Proj(proj_dict)
    sw_x, sw_y = map_proj(grid_dict['sw_lon'], grid_dict['sw_lat'])
    ne_x, ne_y = map_proj(grid_dict['ne_lon'], grid_dict['ne_lat'])
    dx = grid_dict['dx']
    dy = grid_dict['dy']
    if proj_dict['units'] == "m":
        rounding = -2
    else:
        rounding = 0
    x = np.arange(np.round(sw_x, rounding), np.round(ne_x, rounding) + dx, dx)
    y = np.arange(np.round(sw_y, rounding), np.round(ne_y, rounding) + dy, dy)
    x_grid, y_grid = np.meshgrid(x, y)
    lon_grid, lat_grid = map_proj(x_grid, y_grid, inverse=True)
    mapping_data = {'lon': lon_grid, 'lat': lat_grid, 'x': x_grid, 'y': y_grid}
    return mapping_data


def get_proj_obj(proj_dict):
    return Proj(proj_dict)

if __name__ == "__main__":
    main()
