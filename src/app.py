import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gradient_descent import gradient_descent, quadratic, quadratic_gradient, rosenbrock, rosenbrock_gradient, quadratic1, quadratic1_gradient, quadratic2, quadratic2_gradient
from visualizer import plot_contour_and_path

st.title('Gradient Descent Visualizer')

# Sidebar for user inputs
st.sidebar.header('Parameters')
function = st.sidebar.selectbox('Function to minimize', ['Quadratic', 'Quadratic1', 'Quadratic2', 'Rosenbrock'])
learning_rate = st.sidebar.slider('Learning Rate', 0.001, 1.0, 0.1, 0.01)
num_iterations = st.sidebar.slider('Number of Iterations', 10, 1000, 100)
initial_x = st.sidebar.slider('Initial X', -5.0, 5.0, 0.0, 0.1)
initial_y = st.sidebar.slider('Initial Y', -5.0, 5.0, 0.0, 0.1)

# Set up the optimization problem
if function == 'Quadratic':
    f = quadratic
    grad_f = quadratic_gradient
    title = 'Gradient Descent on Quadratic Function'
if function == 'Quadratic1':
    f = quadratic1
    grad_f = quadratic1_gradient
    title = 'Gradient Descent on Quadratic1 Function'
if function == 'Quadratic2':
    f = quadratic2
    grad_f = quadratic2_gradient
    title = 'Gradient Descent on Quadratic2 Function'
else:
    f = rosenbrock
    grad_f = rosenbrock_gradient
    title = 'Gradient Descent on Rosenbrock Function'

# Perform gradient descent
initial_point = [initial_x, initial_y]
path = gradient_descent(f, grad_f, initial_point, learning_rate, num_iterations)

# Create two columns for the plots
col1, col2 = st.columns(2)

# Visualize the results (contour plot)
with col1:
    st.subheader("Contour Plot with Gradient Descent Path")
    fig_contour = plot_contour_and_path(f, path, title)
    st.pyplot(fig_contour)

# Create 3D surface plot
with col2:
    st.subheader("3D Surface Plot of the Function")
    fig_3d = plt.figure(figsize=(10, 8))
    ax = fig_3d.add_subplot(111, projection='3d')

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([f([xi, yi]) for xi, yi in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')
    ax.set_title(f'3D Surface Plot of {function} Function')
    fig_3d.colorbar(surf)

    st.pyplot(fig_3d)

# Display final result
st.write(f'Final point: {path[-1]}')
st.write(f'Final function value: {f(path[-1]):.6f}')