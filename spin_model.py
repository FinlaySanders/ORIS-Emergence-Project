import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer
from spin_dataset import create_k_step_dataset
import matplotlib.pyplot as plt
import time

class NodeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_mlp = Seq(Lin(6, 16), ReLU(), Lin(16, 16), ReLU(), Lin(16, 3)) 

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        row, col = edge_index
        out = scatter(x[row], col, dim=0, dim_size=x.size(0),reduce='sum')
        out = torch.cat([x, out], dim=1)
        out = self.node_mlp(out)

        return out

def train_one_epoch(model, optimizer, loss_fn, loader):
    model.train()
    total_loss = 0

    for batch in loader:
        initial_data_batch = batch[0]
        ground_truths_batches = batch[1:]

        optimizer.zero_grad()

        prediction = initial_data_batch.x
        loss = 0

        for ground_truth in ground_truths_batches:
            prediction, _, _ = model(x=prediction, edge_index=initial_data_batch.edge_index, edge_attr=None, u=None, batch=initial_data_batch.batch)
            loss += loss_fn(prediction, ground_truth.x)
            
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate_model(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            
            initial_data_batch = batch[0]
            ground_truths_batches = batch[1:]
            
            loss = 0
            prediction = initial_data_batch.x
            for sub_batch in ground_truths_batches:
                prediction, _, _ = model(x=prediction, edge_index=initial_data_batch.edge_index)
                loss += loss_fn(prediction, sub_batch.x)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss}")
    return avg_loss

def train():
    model = MetaLayer(node_model=NodeModel())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    print("Creating Dataset...")
    train_loader, val_loader = create_k_step_dataset(size=20, amount=400, depth=200, train_val_split=0.8, k=10)
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

        torch.save(model.node_model.state_dict(), f'Models/NodeModel_E{epoch + 1}.pt')

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