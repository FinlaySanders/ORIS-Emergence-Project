import numpy as np
import XY_to_graph
import torch
from copy import deepcopy

class XY_model:
    def __init__(self, size=20, temp=0.1, spin_model=None, spin_grid=None, spin_vel_grid=None):
        self.J  = 1
        self.temp = temp
        self.size = size

        if spin_grid is not None:
            self.spin_grid = spin_grid
        else:
            self.spin_grid = np.random.rand(size, size) * 2 * np.pi
        if spin_vel_grid is not None:
            self.spin_vel_grid = spin_vel_grid
        else:
            self.spin_vel_grid = np.zeros((size,size))
        
        if spin_model != None:
            self.spin_model = spin_model
            self.x = XY_to_graph.get_xy_spin_node_features(self.spin_grid, self.spin_vel_grid)
            self.edge_index = XY_to_graph.get_xy_edge_index((size,size))
            self.batch = torch.zeros((size*size), dtype=torch.int32)
            self.u = torch.tensor([[1]], dtype=torch.float)
            spin_model.eval()

        self.idxs = np.array([(y, x) for y in range(self.size) for x in range(self.size)])

    # performs a metropolis step on each spin 
    def metropolis_sweep(self):
        np.random.shuffle(self.idxs)
        
        for idx in self.idxs:
            y,x = idx[0],idx[1]

            neighbours = [(y, (x+1) % self.size),(y, (x-1) % self.size),((y-1) % self.size, x),((y+1) % self.size, x)]
            
            prev_theta = self.spin_grid[y,x]
            prev_energy = -self.J*sum(np.cos(prev_theta - self.spin_grid[n]) for n in neighbours) 
            new_theta = (prev_theta + np.random.uniform(-np.pi, np.pi) * 0.1) % (np.pi * 2)
            new_energy = -self.J*sum(np.cos(new_theta - self.spin_grid[n]) for n in neighbours) 
            
            d_energy = new_energy-prev_energy
            if np.random.uniform(0.0, 1.0) < np.exp(-(d_energy/self.temp)):
                self.spin_grid[y,x] = new_theta

    def numerical_integration(self, time):
        dt = 0.05  # smaller time step for improved accuracy
        steps = int(time / dt)

        for _ in range(steps):
            # Update spin grid
            self.spin_grid += dt * self.spin_vel_grid

            # Damping term
            self.spin_vel_grid *= 0.99

            # Keep angles in (0, 2pi)
            self.spin_grid = self.spin_grid % (2 * np.pi)

            # Update spin velocities
            self.spin_vel_grid += -dt * (
                + np.sin(self.spin_grid - np.roll(self.spin_grid, 1, axis=0))
                + np.sin(self.spin_grid - np.roll(self.spin_grid, -1, axis=0))
                + np.sin(self.spin_grid - np.roll(self.spin_grid, 1, axis=1))
                + np.sin(self.spin_grid - np.roll(self.spin_grid, -1, axis=1))
            )

    # returns the indices of vortices and antivortices in the form [y,x] from spins in form (0, 2pi)
    def find_vortices(self):
        # summing acute angles around each potential vortex positions
        sum_angles = (self.get_signed_angles_between(self.spin_grid, np.roll(self.spin_grid, -1, axis=1)) 
                      + self.get_signed_angles_between(np.roll(self.spin_grid, -1, axis=1), np.roll(self.spin_grid, -1, axis=(1,0))) 
                      + self.get_signed_angles_between(np.roll(self.spin_grid, -1, axis=(1,0)), np.roll(self.spin_grid, -1, axis=0)) 
                      + self.get_signed_angles_between(np.roll(self.spin_grid, -1, axis=0), self.spin_grid))

        row, col = np.where(np.isclose(sum_angles, 2*np.pi))
        vortices = list(zip(row+0.5,col+0.5))
        row, col = np.where(np.isclose(sum_angles, -2*np.pi))
        a_vortices = list(zip(row+0.5,col+0.5))

        return vortices, a_vortices
    
    def get_signed_angles_between(self, arr1, arr2):
        diff = arr1 - arr2

        diff[diff>np.pi] = diff[diff>np.pi] % -np.pi
        diff[diff<-np.pi] = diff[diff<-np.pi] % np.pi

        return diff


    # animates the plotted grid's spins using monte carlo and the metropolis algorithm
    def update_spins_monte_carlo(self):
        self.metropolis_sweep()

        U, V = np.cos(self.spin_grid), np.sin(self.spin_grid)
        self.q.set_UVC(U, V)
        return self.q
    
    # animates the plotted grid's spins using numerical integration
    def update_spins_numerical_integration(self):
        self.numerical_integration(1)

        U, V = np.cos(self.spin_grid), np.sin(self.spin_grid)
        self.q.set_UVC(U, V)
        return self.q

    # animates the plotted grid's spins using a trained GNN
    def update_spins_GNN(self, draw=True):
        with torch.no_grad():
            edge_attr = XY_to_graph.get_xy_edge_attr(self.spin_grid, self.edge_index)
            self.x, _, _ = self.spin_model(x=self.x, edge_index=self.edge_index, edge_attr=edge_attr, u=torch.tensor([[1]], dtype=torch.float), batch=torch.tensor([0 for _ in range(self.size*self.size)]))
        sin, cos, vel = np.hsplit(self.x.numpy(), 3)
        self.spin_grid = np.arctan2(sin, cos).reshape(self.size,self.size) % (2*np.pi)

        if not draw:
            return

        U, V = np.cos(self.spin_grid), np.sin(self.spin_grid)
        self.q.set_UVC(U, V)
        return self.q

    # animates the plotted grid's vortices
    def update_vortices(self):
        self.v.remove()
        vortices, a_vortices = self.find_vortices()
        self.v = self.ax.scatter(
            [n[1] for n in vortices] + [n[1] for n in a_vortices], # x values of vortices, anti vortices
            [n[0] for n in vortices] + [n[0] for n in a_vortices], # y values of vortices, anti vortices
            color=["red" for _ in vortices]+["green" for _ in a_vortices]) # corresponding colours of vortex type 
        
        return self.v 
    

    # plots the current vortices and spins   
    def plot_quiver(self, ax, arrow_colour='black', title="XY Model"):
        self.ax = ax
        self.ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # plotting spins on grid
        x = np.arange(self.size)
        y = np.arange(self.size)
        X, Y = np.meshgrid(x, y)
        self.ax.invert_yaxis()
        self.q = self.ax.quiver(X, Y, np.cos(self.spin_grid), np.sin(self.spin_grid), pivot='mid', color=arrow_colour)

        # finding and plotting vortices
        vortices, a_vortices = self.find_vortices()
        self.v = self.ax.scatter(
            [n[1] for n in vortices] + [n[1] for n in a_vortices], # x values of vortices, anti vortices
            [n[0] for n in vortices] + [n[0] for n in a_vortices], # y values of vortices, anti vortices
            color=["red" for _ in vortices]+["green" for _ in a_vortices]) # corresponding colours of vortex type 
