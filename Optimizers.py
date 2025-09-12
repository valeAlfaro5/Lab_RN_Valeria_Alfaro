# %%
import sys
import DnnLib
import numpy as np

# %%
#Example 1

# Create sample data (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)


# Create a neural network: 2 -> 4 -> 1
layer1 = DnnLib.DenseLayer(2, 4, DnnLib.ActivationType.RELU)
layer2 = DnnLib.DenseLayer(4, 1, DnnLib.ActivationType.SIGMOID)

# Create optimizer
optimizer = DnnLib.Adam(learning_rate=0.01)

for epoch in range(100):
    # Forward pass
    h1 = layer1.forward(X)
    output = layer2.forward(h1)
    
    # Compute loss
    loss = DnnLib.mse(output, y)
    
    # Backward pass
    loss_grad = DnnLib.mse_gradient(output, y)
    grad2 = layer2.backward(loss_grad)
    grad1 = layer1.backward(grad2)
    
    # Update parameters
    optimizer.update(layer2)
    optimizer.update(layer1)
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# %%
#Dense Layers

# Basic dense layer: input_dim=3, output_dim=5, default ReLU activation
layer = DnnLib.DenseLayer(3, 5)

# With specific activation function
layer = DnnLib.DenseLayer(3, 5, DnnLib.ActivationType.SIGMOID)

#operations
single_input = np.array([1.0, 2.0, 3.0], dtype=np.float64)
batch_input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)

output = layer.forward(single_input) # Shape: (5,)
batch_output = layer.forward(batch_input) # Shape: (2, 5)

# Forward pass without activation (linear only)
linear_output = layer.forward_linear(single_input)

# Backward pass (for gradient computation)
gradient_input = np.ones(5, dtype=np.float64) # Gradient from next layer
input_gradient = layer.backward(gradient_input)

# Access parameters
weights = layer.weights # Shape: (output_dim, input_dim)
bias = layer.bias # Shape: (output_dim,)

# Access gradients (after backward pass)
weight_grads = layer.weight_gradients
bias_grads = layer.bias_gradients

# Modify activation function
layer.activation_type = DnnLib.ActivationType.TANH

# %%
#binary classification
np.random.seed(42)
x = np.random.randn(1000, 2).astype(np.float64)
y = (x[:,0] + x[:, 1] > 0).astype(np.float64).reshape(-1,1)

print(f"Dataset: {x.shape[0]} samples, {x.shape[1]} features")
print(f"Positive class: {np.sum(y)}/{len(y)} samples")

# %%
layers = [
 DnnLib.DenseLayer(2, 8, DnnLib.ActivationType.RELU),
 DnnLib.DenseLayer(8, 4, DnnLib.ActivationType.RELU),
 DnnLib.DenseLayer(4, 1, DnnLib.ActivationType.SIGMOID)
]

# %%



