'''
minimal reverse-mode automatic differentiation engine: not numerical differentiation nor symbolic differentiation

* Supports vector inputs x = [x0, x1, ...]
* Performs backpropagation through a computation graph
* Includes operators: +, *, sin, cos, pow, exp, log, ReLU
* Visualizes the computation graph using Graphviz

Feature	                    This Code	    PyTorch	    JAX                 SymPy
Reverse-mode autodiff	    ✅ Yes	    ✅ Yes	    ✅ Yes	            ❌ No (symbolic)
Computation graph (dynamic)	✅ Yes	    ✅ Yes	    ❌ (static tracing)	✅ Symbolic AST
Symbolic manipulation	    ❌ No	    ❌ No	    ❌ No	            ✅ Yes
Define-by-run	            ✅ Yes	    ✅ Yes	    ❌ No	            ❌ No
Numerical execution	        ✅ Yes	    ✅ Yes	    ✅ Yes	            ⚠️ Slow or symbolic
GPU support	                ❌ No	    ✅ CUDA	    ✅ XLA	            ❌ No
'''

import math
from graphviz import Digraph
import tempfile  # for temporary file handling

# ===== Node Class for Reverse-mode AD =====
class Node:
    def __init__(self, value, parents=(), op="", label=None):
        self.value = value              # scalar value
        self.grad = 0.0                 # gradient from output node
        self.parents = parents          # (parent_node, local_grad_fn) pairs
        self.op = op                    # operation name
        self.label = label or f"{value:.4f}"  # default to be a 4-decimal value when label such as "sin" is not provided, e.g., a constant
        self.id = id(self)              # unique ID for graphviz

    def backward(self, grad=1.0):
        self.grad += grad
        for parent, local_grad_fn in self.parents:
            parent.backward(grad * local_grad_fn())  # propagate gradient to parents using the lambda function


 # Operator overloads for +, -, *, /, ** etc:
 # create nodes with vals and their parents with grad functions to be evalated later
    def __add__(self, other):  # for addition, e.g., Node(x) + 3
        other = other if isinstance(other, Node) else Node(other)
        return Node(self.value + other.value,
                    parents=((self, lambda: 1.0), (other, lambda: 1.0)),
                    op='+')

    def __radd__(self, other):   # for right addition, e.g., 3 + Node(x)
        return self.__add__(other)

    def __mul__(self, other):  # for multiplication, e.g., Node(x) * 3
        other = other if isinstance(other, Node) else Node(other)
        return Node(self.value * other.value,
                    parents=((self, lambda: other.value), (other, lambda: self.value)),
                    op='*')

    def __rmul__(self, other):    # for right multiplication, e.g., 3 * Node(x)
        return self.__mul__(other)

    def __pow__(self, power):   # for power, e.g., Node(x) ** 2
        if not isinstance(power, (int, float)):
            raise TypeError("Power must be an integer or float")
        return Node(self.value ** power,
                    parents=((self, lambda: power * self.value ** (power - 1)),),
                    op=f'pow({power})')

    def __sub__(self, other):   # for subtraction, e.g., Node(x) - 3
        other = other if isinstance(other, Node) else Node(other)
        return Node(self.value - other.value,
                    parents=((self, lambda: 1.0), (other, lambda: -1.0)),
                    op='-')

    def __rsub__(self, other):  # for right subtraction, e.g., 3 - Node(x)
        other = other if isinstance(other, Node) else Node(other)
        return other.__sub__(self)

    def __truediv__(self, other):  # for division, e.g., Node(x) / 3
        other = other if isinstance(other, Node) else Node(other)
        return Node(self.value / other.value,
                    parents=((self, lambda: 1.0 / other.value),
                             (other, lambda: -self.value / (other.value ** 2))),
                    op='/')

    def __rtruediv__(self, other):  # for right division, e.g., 3 / Node(x)
        other = other if isinstance(other, Node) else Node(other)
        return other.__truediv__(self)


# Override math functions to accept Node objects:
 # create a node with val and its parent with grad functions to be evalated later
def sin(x):
    if isinstance(x, Node):
        # node --> parent: x;  forward value: math.sin(x.value),  local gradient function:  lambda: zero-argument function
        # We use function instead of math.cost(x.value) to avoid immediate evaluation, for example, later for backward pass
        return Node(math.sin(x.value), parents=((x, lambda: math.cos(x.value)),), op='sin')
    else:
        return math.sin(x)

def cos(x):
    if isinstance(x, Node):
        return Node(math.cos(x.value), parents=((x, lambda: -math.sin(x.value)),), op='cos')
    else:
        return math.cos(x)

def exp(x):
    if isinstance(x, Node):
        val = math.exp(x.value)
        return Node(val, parents=((x, lambda: val),), op='exp')
    else:
        return math.exp(x)

def log(x):
    if isinstance(x, Node):
        if x.value <= 0:
            raise ValueError("log(x) requires x > 0")
        return Node(math.log(x.value), parents=((x, lambda: 1/x.value),), op='log')
    else:
        return math.log(x)

def relu(x):
    if isinstance(x, Node):
        val = max(0.0, x.value)
        return Node(val, parents=((x, lambda: 1.0 if x.value > 0 else 0.0),), op='ReLU')
    else:
        return max(0.0, x)

# ===== Graph Visualization =====
def visualize_graph(output_node, filename="autodiff_graph", view=True):
    dot = Digraph(format='png')
    visited = set()

    def trace(node):
        if node.id in visited:
            return
        visited.add(node.id)

        label = f"{node.op}\nval={node.value:.4f}\ngrad={node.grad:.4f}" if node.op else node.label
        dot.node(str(node.id), label)

        for parent, _ in node.parents:
            trace(parent)
            dot.edge(str(parent.id), str(node.id))

    # Start tracing from the output node from which gradients will be computed backward recursveiely
    trace(output_node)
    # Render the graph to a file
    dot.render(filename, view=view)


# ===== Example Function =====
# Example: regular Python function using all supported ops
def regular_func(x):
    # x is a list of Nodes or floats
    return relu(x[0]) + sin(x[0]) * x[1] + exp(x[1]) - log(x[0] + 3)

def simple_multi_func(x):
    # x is a list of Nodes or floats
    return (x[0] * 2) + (5 *x[1]) + (x[0] * x[1])

def simple_add_func(x):
    # x is a list of Nodes or floats
    return (x[0] + 2) + (5 + x[1]) + (x[0] + x[1])


# ===== Main =====
if __name__ == "__main__":

    ### Let's call function with regular input
    x = [-1.5, 2.0]
    y = regular_func(x)  # This will not build graph, just compute value
    print(f"Output value with regular input: {y:.4f}")


    ### Now let's wrap inputs as Node to build graph
    ### Call function normally, but it builds graph with Node op
    x_node = [Node(x[0], label='x[0]'), Node(x[1], label='x[1]')]

    ## Simple add function
    y_add_node = simple_add_func(x_node)
    print(f"Output value: {y_add_node.value:.4f}")

    # Run backward pass to compute gradients
    y_add_node.backward()

    for i, xi in enumerate(x_node):
        print(f"dy/dx[{i}] = {xi.grad:.4f}")

    # Visualize computation graph
    visualize_graph(y_add_node, tempfile.gettempdir() + '/autodiff_add_graph')


    ## Simple multiply function
    y_multi_node = simple_multi_func(x_node)
    print(f"Output value: {y_multi_node.value:.4f}")

    # Run backward pass to compute gradients
    y_multi_node.backward()

    for i, xi in enumerate(x_node):
        print(f"dy/dx[{i}] = {xi.grad:.4f}")

    # Visualize computation graph
    visualize_graph(y_multi_node, tempfile.gettempdir() + "/autodiff_multi_graph")


    ## Regular function
    y_node = regular_func(x_node)
    print(f"Output value: {y_node.value:.4f}")

    # Run backward pass to compute gradients
    y_node.backward()

    for i, xi in enumerate(x_node):
        print(f"dy/dx[{i}] = {xi.grad:.4f}")

    # Visualize computation graph
    visualize_graph(y_node, tempfile.gettempdir() + "/autodiff_regular_graph")



