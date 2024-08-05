import numpy as np

def gradient_descent(f, grad_f, initial_point, learning_rate, num_iterations):
    """
    Perform gradient descent optimization.
    
    Args:
    f: Function to minimize
    grad_f: Gradient of the function
    initial_point: Starting point for optimization
    learning_rate: Step size for each iteration
    num_iterations: Number of iterations to perform
    
    Returns:
    List of points visited during optimization
    """
    point = np.array(initial_point, dtype=float)
    path = [point]
    
    for _ in range(num_iterations):
        gradient = grad_f(point)
        point = point - learning_rate * gradient
        path.append(point.copy())
    
    return np.array(path)

# Example functions
def quadratic(x):
    return x[0]**2 + x[1]**2

def quadratic_gradient(x):
    return np.array([2*x[0], 2*x[1]])

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_gradient(x):
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])

# Add more functions as needed