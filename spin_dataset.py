from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
import random

from XY import XY_model
import XY_to_graph

def create_k_step_dataset(size, amount, depth, train_val_split=0.8, k=10):
    sequences = create_data_sequences(size, amount, depth)

    # batching subsequences by time 
    sub_sequences = []

    for sequence in sequences:
        for i in range(0, depth - k, k):
            sub_sequence = sequence[i:i+k]
            sub_sequences.append(sub_sequence)
    
    random.shuffle(sub_sequences)

    split_idx = int(train_val_split * len(sub_sequences))
    train_sequences = sub_sequences[:split_idx]
    val_sequences = sub_sequences[split_idx:]

    train_loader = DataLoader(train_sequences, batch_size=5, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_sequences, batch_size=5, shuffle=True, collate_fn=custom_collate)
    
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


def create_data_sequences(size, amount, length):
    sequences = []

    edge_index = XY_to_graph.get_xy_edge_index((size, size))

    for i in range(amount):
        new_sequence = []
        xy = XY_model(size)

        for _ in range(length):
            x = XY_to_graph.get_xy_spin_node_features(xy.spin_grid, xy.spin_vel_grid)

            new_sequence.append(Data(x=x, edge_index=edge_index))

            xy.numerical_integration(1)
        
        sequences.append(new_sequence)
    
        print(f"Created Sequence {i} of {amount}")
    
    return sequences


if __name__ == '__main__':
    t, v = create_k_step_dataset(5, 100, 100)
    for batch in t:
        print(batch)