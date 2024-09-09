import matplotlib.pyplot as plt
import numpy as np
import math

def bezier_curve(points, t_values):
    n = len(points) - 1
    curve = np.zeros((len(t_values), 2))

    for i, t in enumerate(t_values):
        for j, point in enumerate(points):
            curve[i] += point * (math.comb(n, j) * (1 - t) ** (n - j) * t ** j)
    return curve

def unwrap_trajectory_1d(trajectory, world_size, threshold=None):
    if threshold is None:
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

def unwrap_trajectory_2d(trajectory, world_size, threshold=None):
    x_unwrapped = unwrap_trajectory_1d(trajectory[:, 0], world_size[0], threshold)
    y_unwrapped = unwrap_trajectory_1d(trajectory[:, 1], world_size[1], threshold)
    return np.vstack((x_unwrapped, y_unwrapped)).T

def smooth_trajectory(positions, world_size):
    unwrapped_positions = unwrap_trajectory_2d(positions, (10, 10))
    # Compute the smoothed positions using Bezier curve
    t_values = np.linspace(0, 1, len(positions))
    smoothed_positions = bezier_curve(unwrapped_positions, t_values)
    

    # Plotting original, unwrapped and smoothed trajectories
    plt.figure(figsize=(15, 5))

    # Before Unwrapping
    ax1 = plt.subplot(1, 4, 1)
    plt.plot(positions[:, 0], positions[:, 1], 'o', label='Before Unwrapping')
    plt.title('Original Trajectory')
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    ax1.set_xticks([])
    ax1.set_yticks([])


    # After Unwrapping
    ax2 = plt.subplot(1, 4, 2)
    plt.plot(unwrapped_positions[:, 0], unwrapped_positions[:, 1], 'o', label='After Unwrapping')
    plt.title('Unwrapped Trajectory')
    ax2.set_xticks([])
    ax2.set_yticks([])


    # Smoothed
    ax3 = plt.subplot(1, 4, 3)
    plt.plot(smoothed_positions[:, 0], smoothed_positions[:, 1], 'o', color='red', label='Smoothed by Bezier')
    plt.title('Smoothed Trajectory')
    ax3.set_xticks([])
    ax3.set_yticks([])

    # wrapped smooth
    ax4 = plt.subplot(1, 4, 4)
    plt.plot(smoothed_positions[:, 0] % world_size[0], smoothed_positions[:, 1] % world_size[1], 'o', color='red', label='Smoothed by Bezier')
    plt.title('Final Trajectory')
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    ax4.set_xticks([])
    ax4.set_yticks([])


    plt.tight_layout()
    plt.show()
    
    return smoothed_positions

# Example usage
positions = np.array([
    [1,1],
    [2,2],
    [3,2],
    [3,1],
    [4,0],
    [5,10],
    [6,8],
    [7,8],
    [8,7],
    [9,6],
    [10,6],
    [1,5],
    [2,4],
    [3,4],
    [4,3],
    [5,2],
    [6,1],
    [7,2],
    ]
)

print("n positions: ", len(positions))
print(positions)
smoothed_positions = smooth_trajectory(positions[::-1], (10, 10))
print("n smoothed positions: ", len(smoothed_positions))
print(smoothed_positions)