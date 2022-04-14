from hagelslag.processing.STObject import STObject
import numpy as np


def test_STObject_creation():
    data = np.array([[0, 0, 0, 0],
                     [0, 1.2, 1, 0],
                     [0, 5, 0, 0]])
    mask = np.where(data > 0, 1, 0).astype(int)
    print("Mask", mask)
    print("Mask shape", mask.shape, data.shape)
    full_grid_shape = (700, 400)
    dx = 3000.0
    false_easting = 30000.0
    false_northing = 50000.0
    patch_radius = 16
    i, j = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    i += 100
    j += 200
    i_full, j_full = np.meshgrid(np.arange(full_grid_shape[1]), np.arange(full_grid_shape[0]))
    x = dx * j + false_easting
    y = dx * i + false_northing
    x_full = dx * i_full + false_easting
    y_full = dx * j_full + false_northing
    sto = STObject(data, mask, x, y, i, j, 10, 10, dx=dx)
    center_i, center_j = sto.center_of_mass_ij(10)
    assert (center_i >= i.min()) and (center_i <= i.max())
    assert (center_j >= j.min()) and (center_j <= j.max())
    print(x_full.shape, y_full.shape)
    print(i_full.shape, j_full.shape)
    patch_sto = sto.extract_patch(patch_radius, x_full, y_full, i_full, j_full)
    assert patch_sto.timesteps[0].shape[0] == patch_radius * 2, patch_sto.timesteps[0].shape[0]
    return