import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gradient_descent import gradient_descent, quadratic, quadratic_gradient, rosenbrock, rosenbrock_gradient, quadratic1, quadratic1_gradient, himmelblau, himmelblau_gradient
from visualizer import plot_contour_and_path, plot_surface

st.title('Gradient Descent Visualizer')

# Sidebar for user inputs
st.sidebar.header('Parameters')
function = st.sidebar.selectbox('Function to minimize', ['Quadratic', 'Quadratic1', 'Himmelblau', 'Rosenbrock'])
learning_rate = st.sidebar.slider('Learning Rate', 0.001, 1.0, 0.1, 0.001)
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
if function == 'Himmelblau':
    f = himmelblau
    grad_f = himmelblau_gradient
    title = 'Gradient Descent on Himmelblau Function'
if function == 'Rosenbrock':
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
    title = 'Surface plot of {} Function'.format(function)
    fig_3d = plot_surface(f, title, x_range = (-5, 5), y_range = (-5, 5), num_points = 100)
    st.pyplot(fig_3d)

# Display final result
st.write(f'Final point: {path[-1]}')
st.write(f'Final function value: {f(path[-1]):.6f}')