import streamlit as st
import numpy as np
from gradient_descent import gradient_descent, quadratic, quadratic_gradient, rosenbrock, rosenbrock_gradient
from visualizer import plot_contour_and_path

st.title('Gradient Descent Visualizer')

# Sidebar for user inputs
st.sidebar.header('Parameters')
function = st.sidebar.selectbox('Function to minimize', ['Quadratic', 'Rosenbrock'])
learning_rate = st.sidebar.slider('Learning Rate', 0.001, 1.0, 0.1, 0.001)
num_iterations = st.sidebar.slider('Number of Iterations', 10, 1000, 100)
initial_x = st.sidebar.slider('Initial X', -5.0, 5.0, 0.0, 0.1)
initial_y = st.sidebar.slider('Initial Y', -5.0, 5.0, 0.0, 0.1)

# Set up the optimization problem
if function == 'Quadratic':
    f = quadratic
    grad_f = quadratic_gradient
    title = 'Gradient Descent on Quadratic Function'
else:
    f = rosenbrock
    grad_f = rosenbrock_gradient
    title = 'Gradient Descent on Rosenbrock Function'

# Perform gradient descent
initial_point = [initial_x, initial_y]
path = gradient_descent(f, grad_f, initial_point, learning_rate, num_iterations)

# Visualize the results
fig = plot_contour_and_path(f, path, title)
st.pyplot(fig)

# Display final result
st.write(f'Final point: {path[-1]}')
st.write(f'Final function value: {f(path[-1]):.6f}')