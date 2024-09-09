import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer
from vortex_dataset import create_data_sequences, create_k_step_dataset
import matplotlib.pyplot as plt
import time

class NodeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_mlp = Seq(Lin(10, 16), ReLU(), Lin(16, 16), ReLU(), Lin(16, 5)) 

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        non_zero_mask = torch.any(x != 0, dim=1)
        num_non_zero = non_zero_mask.sum().float()
        num_non_zero = num_non_zero if num_non_zero > 0 else 1

        row, col = edge_index
        out = scatter(x[row], col, dim=0, dim_size=x.size(0), reduce='sum')
        out = out / num_non_zero

        out = torch.cat([x, out], dim=1)
        out = self.node_mlp(out)

        return out

def train_one_epoch(model, optimizer, loss_fn, loader):
    model.train()
    total_loss = 0.0

    for batch in loader:
        # include all vortices in training
        mask = [True for _ in range(len(batch[0].x))]  
        initial_data_batch = batch[0]
        ground_truths_batches = batch[1:]

        optimizer.zero_grad()

        prediction = initial_data_batch.x
        loss = 0

        for ground_truth in ground_truths_batches:
            prediction, _, _ = model(x=prediction, edge_index=initial_data_batch.edge_index, edge_attr=None, u=None, batch=initial_data_batch.batch)
            
            for i, x in enumerate(ground_truth.x):
                if x[4] == 0:
                    mask[i] = False
                
            loss += loss_fn(prediction[mask], ground_truth.x[mask])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate_model(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            # include all vortices in training initially 
            mask = [True for _ in range(len(batch[0].x))]  
            initial_data_batch = batch[0]
            ground_truths_batches = batch[1:]

            prediction = initial_data_batch.x
            loss = 0

            for ground_truth in ground_truths_batches:
                prediction, _, _ = model(x=prediction, edge_index=initial_data_batch.edge_index, edge_attr=None, u=None, batch=initial_data_batch.batch)
                
                for i, x in enumerate(ground_truth.x):
                    if x[4] == 0:
                        mask[i] = False
                    
                loss += loss_fn(prediction[mask], ground_truth.x[mask])
            
        total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss}")
    return avg_loss

def train():
    model = MetaLayer(node_model=NodeModel())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    lattice_size = 20

    print("Loading Dataset")

    sequences = create_data_sequences(lattice_size, amount=1000, max_depth=200, cooling=15)
    train_loader, val_loader = create_k_step_dataset(sequences, train_val_split=0.8, k=5, batch_size=5)

    print("Done!")

    train_losses = []
    val_losses = []

    n_epochs = 10
    for epoch in range(n_epochs):
        start_time = time.time()

        avg_train_loss = train_one_epoch(model, optimizer, loss_fn, train_loader)
        train_losses.append(avg_train_loss)
        
        avg_val_loss = validate_model(model, val_loader, loss_fn)
        val_losses.append(avg_val_loss)

        torch.save(model.node_model.state_dict(), f'Models/TESTVortexModel_1000x200_2x16_k5_b5_E{epoch + 1}.pt')

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}/{n_epochs} - Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Time: {epoch_duration:.2f} seconds")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses over Time')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train()




