import test
import random
import sys
import tempfile   # For storing results in tmp directory
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever need
from python_algo_production.algo.mini_autodiff import *

# Auotograde does not inclue RELU
def autograd_relu(x):
    return np.maximum(0, x)

### Basic function testing
def basic_func(x):
    # x is a list of Nodes or floats
    return relu(x[0]) + sin(x[0]) * x[1] + exp(x[1]) - log(x[0] + 3)

def basic_autograd_func(x):  # we must use np.exp etc for autograd to work correctly
    # x is a list of np floats
    return autograd_relu(x[0]) + np.sin(x[0]) * x[1] + np.exp(x[1]) - np.log(x[0] + 3)

def test_mini_autodiff_basic():
    x = [-1.5, 2.0]   # the input values for the function as a list
    # Now let's wrap inputs as Node to build graph
    x_node = [Node(x[0], label='x[0]'), Node(x[1], label='x[1]')]

    # Call function normally, but it builds graph with Node ops
    y_node = basic_func(x_node)
    print(f"Output value: {y_node.value:.4f}")
    # Run backward pass to compute gradients
    y_node.backward()
    # create a vector of gradients using lambda
    # Collect gradients into a list
    grad_vector = [xi.grad for xi in x_node]
    print(f"MiniAutoDiff Gradients: {grad_vector}")
    # Visualize computation graph
    visualize_graph(y_node, tempfile.gettempdir() + "/simple_mini_autodiff_graph")

    # ground truth gradients
    autograd_testfunc = grad(basic_autograd_func)       # Obtain its gradient function
    autograd_vector = autograd_testfunc(x)  # Evaluate the gradient at x
    print(f"Autograd gradients: {autograd_vector}")

    # Let's compare the result with given tolerance
    assert np.allclose(grad_vector, autograd_vector, atol=1e-5), "Gradients do not match!"


### Complex function testing
def complex_func(x):
    # x is a list of Nodes or floats
    return relu(x[0] + sin(x[0]) * x[1] + exp(x[1] - log(x[0] + 3)))

def complex_autograd_func(x):  # we must use np.exp etc for autograd to work correctly
    # x is a list of np floats
    return autograd_relu(x[0] + np.sin(x[0]) * x[1] + np.exp(x[1] - np.log(x[0] + 3)))

def test_mini_autodiff_complex():
    x = [-1.5, 2.0]   # the input values for the function as a list
    # Now let's wrap inputs as Node to build graph
    x_node = [Node(x[0], label='x[0]'), Node(x[1], label='x[1]')]

    # Call function normally, but it builds graph with Node ops
    y_node = complex_func(x_node)
    print(f"Output value: {y_node.value:.4f}")
    # Run backward pass to compute gradients
    y_node.backward()
    # create a vector of gradients using lambda
    # Collect gradients into a list
    grad_vector = [xi.grad for xi in x_node]
    print(f"MiniAutoDiff Gradients: {grad_vector}")
    # Visualize computation graph
    visualize_graph(y_node, tempfile.gettempdir() + "/complex_mini_autodiff_graph")

    # ground truth gradients
    autograd_testfunc = grad(complex_autograd_func)       # Obtain its gradient function
    autograd_vector = autograd_testfunc(x)  # Evaluate the gradient at x
    print(f"Autograd gradients: {autograd_vector}")

    # Let's compare the result with given tolerance
    assert np.allclose(grad_vector, autograd_vector, atol=1e-5), "Gradients do not match!"
