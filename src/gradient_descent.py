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
        path.append(point)
    
    return np.array(path)

# Example functions
def quadratic(x):
    return x[0]**2 + x[1]**2

def quadratic_gradient(x):
    return np.array([2*x[0], 2*x[1]])

def quadratic1(x):
    return x[0]**2 + 10*x[1]**2

def quadratic1_gradient(x):
    return np.array([2*x[0], 20*x[1]])

def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def himmelblau_gradient(x):
    dfdx = 4 * x[0] * (x[0]**2 + x[1] - 11) + 2 * (x[0] + x[1]**2 - 7)
    dfdy = 2 * (x[0]**2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1]**2 - 7)
    return np.array([dfdx, dfdy])

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_gradient(x):
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])