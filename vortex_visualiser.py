import torch
import XY_to_graph

class Vortex_Visualiser:
    def __init__(self, size, vortex_poses, avortex_poses, vortex_model):
        vortex_data = []

        for pos in vortex_poses:
            vortex_data.append(XY_to_graph.pos_to_angles(pos, size) + (1,))
            #vortex_data.append((pos[0]/size, pos[1]/size, 1))
        for pos in avortex_poses:
            vortex_data.append(XY_to_graph.pos_to_angles(pos, size) + (-1,))
            #vortex_data.append((pos[0]/size, pos[1]/size, -1))

        self.vortex_data = torch.tensor(vortex_data, dtype=torch.float)
    
        self.vortex_model = vortex_model
        self.size = size
    
    def update_vortices(self):
        n_vortices = len(self.vortex_data)
        if n_vortices == 0:
            self.ax.clear()
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.ax.set_xlim(0, self.size)
            self.ax.set_ylim(0, self.size)  
            self.ax.set_title("Predicted Vortices")
            self.ax.invert_yaxis()
            return

        rows, cols = torch.combinations(torch.arange(n_vortices), 2).t()
        edge_index = torch.cat([torch.stack([rows, cols]), torch.stack([cols, rows])], dim=1)
        with torch.no_grad():
            self.vortex_data, _, _ = self.vortex_model(x=self.vortex_data, edge_index=edge_index, batch=torch.tensor([0 for _ in range(n_vortices)]))

        self.vortex_data[:, 4] = self.vortex_data[:, 4].round()
        #self.vortex_data[:, 2] = self.vortex_data[:, 2].round()
        #self.vortex_data[:, 0:2] = self.vortex_data[:, 0:2] % 1

        vortices = self.vortex_data_to_vortices()
        #vortices = self.vortex_data
        self.vortex_data = self.vortex_data[self.filter_annihilations(vortices)]
        
        self.plot_vortices(vortices)

    def plot_real_vortices(self, real_vortices):
        v, av = real_vortices

        x = [[pos[1]] for pos in v] + [[pos[1]] for pos in av]
        y = [[pos[0]] for pos in v] + [[pos[0]] for pos in av]
        colors = ["red" for _ in range(len(v))] + ["green" for _ in range(len(av))]

        self.ax.scatter(x, y, color=colors, alpha=0.5, s=50)

    def plot_scatter(self, ax, title="Predicted Vortices"):
        self.ax = ax 
        ax.set_title(title)
        vortices = self.vortex_data_to_vortices()
        #vortices = self.vortex_data
        self.plot_vortices(vortices)

    def plot_vortices(self, vortices):
        self.ax.clear()
        x = []
        y = []
        colors = []
        # stored as row, col
        for vortex in vortices:
            x.append(vortex[1])
            y.append(vortex[0])
            if vortex[2] > 0:
                colors.append("red")
            else:
                colors.append("green")
        
        self.ax.scatter(x, y, color=colors, s=50)
        
        self.ax.set_title("Predicted Vortices")
        self.ax.set_xlim(0, self.size)  
        self.ax.set_ylim(0, self.size)  
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.invert_yaxis()
    
    def vortex_data_to_vortices(self):
        euclidian_vortex_data = []
        for pos in self.vortex_data:
            euclidian_vortex_data.append(XY_to_graph.angles_to_pos(pos, self.size))
        return euclidian_vortex_data
    
    def filter_annihilations(self, positions_list):
        tensor = torch.tensor(positions_list)
        threshold_distance = 2.5 # Define what "close enough" means in terms of distance

        distances = self.pairwise_distance_pbc(tensor[:, :2], self.size, self.size)
        mask_different_n = (tensor[:, 2].unsqueeze(1) != tensor[:, 2].unsqueeze(0))

        # Identify pairs that are close and have different n values
        mask_to_remove = (distances < threshold_distance) & mask_different_n

        # For each row, see if there's any pair that requires its removal
        return ~mask_to_remove.any(dim=1)
    
    def pairwise_distance_pbc(self, matrix, Lx, Ly):
        """Compute the pairwise distance matrix for a 2D torch tensor with PBC."""
        diff = matrix.unsqueeze(1) - matrix.unsqueeze(0)
        
        # Adjust for periodic boundary conditions in x dimension
        diff[..., 0] = torch.where(diff[..., 0] > 0.5 * Lx, diff[..., 0] - Lx, diff[..., 0])
        diff[..., 0] = torch.where(diff[..., 0] < -0.5 * Lx, diff[..., 0] + Lx, diff[..., 0])
        
        # Adjust for periodic boundary conditions in y dimension
        diff[..., 1] = torch.where(diff[..., 1] > 0.5 * Ly, diff[..., 1] - Ly, diff[..., 1])
        diff[..., 1] = torch.where(diff[..., 1] < -0.5 * Ly, diff[..., 1] + Ly, diff[..., 1])
        
        dist = torch.sum(diff ** 2, dim=-1).sqrt()
        return dist

