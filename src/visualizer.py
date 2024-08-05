import numpy as np
import matplotlib.pyplot as plt

def plot_contour_and_path(f, path, title, x_range=(-5, 5), y_range=(-5, 5), num_points=100):
    """
    Plot the contour of the function and the optimization path.
    
    Args:
    f: Function to visualize
    path: List of points visited during optimization
    title: Title of the plot
    x_range, y_range: Range for x and y axes
    num_points: Number of points to use for contour plot
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    Z = np.array([f([X[i, j], Y[i, j]]) for i in range(num_points) for j in range(num_points)]).reshape(num_points, num_points)

    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=50)
    plt.colorbar(label='Function Value')
    plt.plot(path[:, 0], path[:, 1], 'ro-', linewidth=1.5, markersize=3)
    plt.plot(path[0, 0], path[0, 1], 'go', markersize=8, label='Start')
    plt.plot(path[-1, 0], path[-1, 1], 'bo', markersize=8, label='End')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    return plt

def plot_surface(f, title, x_range = (-5, 5), y_range = (-5, 5), num_points = 100):
    fig_3d = plt.figure(figsize=(10, 8))
    ax = fig_3d.add_subplot(111, projection='3d')

    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    Z = np.array([f([xi, yi]) for xi, yi in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')
    ax.set_title(title)
    fig_3d.colorbar(surf)
    return fig_3d
