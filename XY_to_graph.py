import torch
import numpy as np
from math import sin, cos

def get_xy_edge_index(shape):
    idx_grid = np.arange(0,shape[0]*shape[1]).reshape(shape)

    neighbour_arr = (np.stack([
                            np.roll(idx_grid, -1, axis=0),
                            np.roll(idx_grid, 1, axis=0),
                            np.roll(idx_grid, -1, axis=1),
                            np.roll(idx_grid, 1, axis=1)
                           ], axis = 2)).flatten()

    source_arr = np.repeat(idx_grid.flatten(), 4)
    np_edge_index = np.stack((source_arr, neighbour_arr))

    edge_index = torch.from_numpy(np_edge_index).to(torch.int64)
        
    return edge_index

def get_xy_edge_attr(grid, edge_index):
    flat_grid = grid.flatten()
    edge_attr = torch.tensor(np.expand_dims(np.sin(flat_grid[edge_index[0]] - flat_grid[edge_index[1]]), axis=1), dtype=torch.float)
    return edge_attr

def get_xy_spin_node_features(grid, vel_grid):
    x = grid.flatten()
    v = vel_grid.flatten()
    x = np.stack((np.sin(x), np.cos(x), v), axis=1)
    
    return torch.tensor(x, dtype=torch.float)

def pos_to_angles(pos, lattice_size):
    x = pos[1] / lattice_size * 2*torch.pi
    y = pos[0] / lattice_size * 2*torch.pi

    return (sin(x), cos(x), sin(y), cos(y))

def angles_to_pos(angles, lattice_size):
    #angles = torch.tensor(angles)

    # Using atan2 to get the angles in [-pi, pi] range
    x_angle = torch.atan2(angles[0], angles[1])
    y_angle = torch.atan2(angles[2], angles[3])

    # Convert angles back to the [0, lattice_size] range
    x = x_angle / (2 * torch.pi) * lattice_size
    y = y_angle / (2 * torch.pi) * lattice_size

    # Wrap x and y
    x = x % lattice_size
    y = y % lattice_size

    return (y, x, angles[4])


if __name__ == '__main__':    
    #pos = (0, 0)
    #x = pos_to_angles(pos, 30)
    #print(x)
    #y = angles_to_pos(x + (1,), 30)
    #print(y)

    x = (30) / 30 * 2*torch.pi
    y = (0) / 30 * 2*torch.pi
    print(cos(x) - cos(y))


    

