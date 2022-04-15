from hagelslag.processing.STObject import STObject
import numpy as np
from pyproj import Proj
import json

def test_STObject_creation():
    data = np.array([[0, 0, 0, 0],
                     [0, 1.2, 1, 0],
                     [0, 5, 0, 0]])
    mask = np.where(data > 0, 1, 0).astype(int)
    full_grid_shape = (700, 400)
    dx = 3000.0
    false_easting = 30000.0
    false_northing = 50000.0
    patch_radius = 16
    time = 10
    i, j = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    i += 100
    j += 200
    i_full, j_full = np.meshgrid(np.arange(full_grid_shape[1]), np.arange(full_grid_shape[0]))
    x = dx * j + false_easting
    y = dx * i + false_northing
    x_full = dx * i_full + false_easting
    y_full = dx * j_full + false_northing
    sto = STObject(data, mask, x, y, i, j, time, time, dx=dx)
    center_i, center_j = sto.center_of_mass_ij(time)
    assert (center_i >= i.min()) and (center_i <= i.max())
    assert (center_j >= j.min()) and (center_j <= j.max())
    patch_sto = sto.extract_patch(patch_radius, x_full, y_full, i_full, j_full)
    assert patch_sto.timesteps[0].shape[0] == patch_radius * 2, patch_sto.timesteps[0].shape[0]
    boundary_coords = sto.boundary_contour(time)
    assert boundary_coords.shape[0] == 2
    assert (boundary_coords[0].min() >= sto.x[0].min()) & (boundary_coords[0].max() <= sto.x[0].max())
    assert (boundary_coords[1].min() >= sto.y[0].min()) & (boundary_coords[1].max() <= sto.y[0].max())
    proj = Proj(proj="lcc", lat_0=30, lon_0=-96, lat_1=31, lat_2=50)
    json_obj = sto.to_geojson_feature(proj, output_grids=True)
    json_str = json.dumps(json_obj)
    return