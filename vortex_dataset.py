import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from XY import XY_model
import XY_to_graph
from scipy.optimize import linear_sum_assignment
import numpy as np
import math
import random

def create_k_step_dataset(sequences, train_val_split=0.8, k=10, batch_size=5):
    #sequences = create_data_sequences(size, amount, depth, cooling=cooling)
    sub_sequences = []

    for sequence in sequences:
        for i in range(0, len(sequence) - k, k):
            sub_sequence = sequence[i:i+k]
            sub_sequences.append(sub_sequence)
    
    random.shuffle(sub_sequences)

    split_idx = int(train_val_split * len(sub_sequences))
    train_sequences = sub_sequences[:split_idx]
    val_sequences = sub_sequences[split_idx:]

    train_loader = DataLoader(train_sequences, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_sequences, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    return train_loader, val_loader
    
def custom_collate(batch_list):
    # batch_list is a list of sequences from sequences
    # Transpose to group by timestep
    timesteps = list(zip(*batch_list))

    batched_data = []
    for timestep_data in timesteps:
        batch = Batch.from_data_list(timestep_data)
        batched_data.append(batch)

    return batched_data

def create_data_sequences(size, amount, max_depth, cooling=20):
    sequences = []

    for i in range(amount):
        print(f"Creating Sequence {i}  /  {amount}")
        new_sequence = []

        v_traj, av_traj = generate_xy_trajectories(size, max_depth, cooling=cooling)
        v_traj = [smooth_trajectory(traj, size) for traj in v_traj] 
        av_traj = [smooth_trajectory(traj, size) for traj in av_traj]

        # padding sequences for iterative refinement 
        real_max_depth = 0
        for traj in v_traj:
            depth = len(traj)
            if depth > real_max_depth:
                real_max_depth = depth
        
        if real_max_depth == 0:
            continue
        
        for traj in v_traj:
            for i in range(real_max_depth - len(traj)):
                #traj.append(None)
                traj.append(traj[-1])
        for traj in av_traj:
            for i in range(real_max_depth - len(traj)):
                #traj.append(None)
                traj.append(traj[-1])

        # edit below according to padding
        for i in range(real_max_depth - 1):
            x = []
            y = []
            
            for traj in v_traj:
                if traj[i] is not None:
                    x.append(XY_to_graph.pos_to_angles(traj[i], size) + (1,))
                else:
                    x.append([0,0,0,0,0])
            
            for traj in av_traj:
                if traj[i] is not None:
                    x.append(XY_to_graph.pos_to_angles(traj[i], size) + (-1,))
                else:
                    x.append([0,0,0,0,0])

            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)

            rows, cols = torch.combinations(torch.arange(len(x)), 2).t()
            edge_index = torch.cat([torch.stack([rows, cols]), torch.stack([cols, rows])], dim=1)

            new_sequence.append(Data(x=x, y=y, edge_index=edge_index))
            
        sequences.append(new_sequence)
    
    return sequences

# this can be optimised a lot!
def generate_xy_trajectories(size, max_depth, cooling=20):
    xy = XY_model(size)
    for _ in range(cooling):
        xy.numerical_integration(1)
    
    v_trajectories = []
    av_trajectories = []

    prev_v, prev_av = xy.find_vortices()
    for pos in prev_v:
        v_trajectories.append([pos])
    for pos in prev_av:
        av_trajectories.append([pos])

    for depth in range(max_depth):
        xy.numerical_integration(1)

        new_v, new_av = xy.find_vortices()

        #if new_v == [] or new_av == []:
        #    print(f"ran out after{depth+1}")

        v_pairs = pair_vortices(prev_v, new_v, size)
        av_pairs = pair_vortices(prev_av, new_av, size)

        annihilated_vortices = list(set(prev_v) - set(new_v))
        annihilated_avortices = list(set(prev_av) - set(new_av))
        annihilated_pairs = pair_vortices(annihilated_avortices, annihilated_vortices, size)
        annihilation_poses = {}

        for av, v in annihilated_pairs.items():
            mid = pbc_average([av, v], size)
            annihilation_poses[v] = mid
            annihilation_poses[av] = mid
        
        for traj in v_trajectories:
            if len(traj) - 1 != depth:
                continue 
            if traj[-1] in v_pairs:
                traj.append(v_pairs[traj[-1]]) 
            elif traj[-1] in annihilation_poses:
                traj.append(annihilation_poses[traj[-1]]) 
            
        for traj in av_trajectories:
            if len(traj) -1 != depth:
                continue 
            if traj[-1] in av_pairs:
                traj.append(av_pairs[traj[-1]]) 
            elif traj[-1] in annihilation_poses:
                traj.append(annihilation_poses[traj[-1]]) 

        prev_v, prev_av = new_v, new_av
    
    return v_trajectories, av_trajectories

def bezier_curve(points, t_values):
    n = len(points) - 1
    curve = np.zeros((len(t_values), 2))

    for i, t in enumerate(t_values):
        for j, point in enumerate(points):
            curve[i] += point * (math.comb(n, j) * (1 - t) ** (n - j) * t ** j)

    return curve

def unwrap_trajectory_1d(trajectory, world_size):
    threshold = world_size / 2

    unwrapped = [trajectory[0]]
    offset = 0
    for i in range(1, len(trajectory)):
        delta = trajectory[i] - trajectory[i-1]
        if delta > threshold:
            offset -= world_size
        elif delta < -threshold:
            offset += world_size
        unwrapped.append(trajectory[i] + offset)

    return np.array(unwrapped)

def unwrap_trajectory_2d(trajectory, world_size):
    x_unwrapped = unwrap_trajectory_1d(trajectory[:, 0], world_size)
    y_unwrapped = unwrap_trajectory_1d(trajectory[:, 1], world_size)

    return np.vstack((x_unwrapped, y_unwrapped)).T

def smooth_trajectory(positions, world_size):
    positions = np.array(positions)
    positions = unwrap_trajectory_2d(positions, world_size)
    
    t_values = np.linspace(0, 1, len(positions))
    smoothed_positions = bezier_curve(positions, t_values)

    smoothed_positions %= world_size
    
    return smoothed_positions.tolist()

def pair_vortices(prev_v, new_v, lattice_size):
    pairs = {}
    dist_matrix = np.zeros((len(prev_v), len(new_v)))

    for i in range(len(prev_v)):
        for j in range(len(new_v)):
            dist_matrix[i][j] = pbc_distance(prev_v[i], new_v[j], lattice_size)

    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    pairs = {prev_v[i]: new_v[j] for i, j in zip(row_ind, col_ind)}

    return pairs

def pbc_distance(pos1, pos2, lattice_size):
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    
    dx = min(dx, lattice_size - dx)
    dy = min(dy, lattice_size - dy)
    
    return (dx**2 + dy**2)**0.5

def pbc_average(points, size):
    sum_cos_x = 0
    sum_sin_x = 0
    sum_cos_y = 0
    sum_sin_y = 0

    for point in points:
        sum_cos_x += math.cos(2 * math.pi * point[0] / size)
        sum_sin_x += math.sin(2 * math.pi * point[0] / size)
        sum_cos_y += math.cos(2 * math.pi * point[1] / size)
        sum_sin_y += math.sin(2 * math.pi * point[1] / size)

    avg_cos_x = sum_cos_x / len(points)
    avg_sin_x = sum_sin_x / len(points)
    avg_cos_y = sum_cos_y / len(points)
    avg_sin_y = sum_sin_y / len(points)
    
    avg_angle_x = math.atan2(avg_sin_x, avg_cos_x)
    avg_angle_y = math.atan2(avg_sin_y, avg_cos_y)
    
    avg_x = (avg_angle_x / (2 * math.pi)) * size
    avg_y = (avg_angle_y / (2 * math.pi)) * size

    avg_x = avg_x % size
    avg_y = avg_y % size

    return avg_x, avg_y

if __name__ == '__main__':    
    train_loader, val_loader = create_k_step_dataset(size=30, amount=10, depth=50, k=4)
    print("")
    print("Train Loader")
    print("")
    for batch in train_loader:
        print(batch)