# %%
import numpy as np
import DnnLib


# %%
x = np.array([[0.5,-0.2, 0.1]])

# %%
layer = DnnLib.DenseLayer(3, 2, DnnLib.ActivationType.RELU)

# %%
layer.weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
layer.bias = np.array([0.01, -0.02])

# %%
y = layer.forward(x)
print("Salida con activación:", y)

# %%
y_lin = layer.forward_linear(x)
print("Salida lineal:", y_lin)

# %%
print("Sigmoid:", DnnLib.sigmoid(np.array([0.0, 2.0,-1.0])))


