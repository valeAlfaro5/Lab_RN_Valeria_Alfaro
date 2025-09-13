# %%
import numpy as np
import DnnLib
import json
import matplotlib.pyplot as plt

# %%
#carga de datos
with open("mnist_mlp_pretty.json", "r") as f:
    datos = json.load(f)

#carga imagenes y labels
data = np.load("mnist_test.npz")
images = data["images"]
labels = data["labels"]

#capa1
w1 = np.array(datos["layers"][0]["W"])
b1 = np.array(datos["layers"][0]["b"])

#capa2
w2 = np.array(datos["layers"][1]["W"])
b2 = np.array(datos["layers"][1]["b"])

#activación por capa
activate1 = datos["layers"][0]["activation"]
activate2 = datos["layers"][1]["activation"]

# %%
print(w1.shape, b1.shape, w2.shape, b2.shape)

# %%
#definir capas densas (ambas son densas segun el json)
#primera capa tiene 784 entradas y 128 salidas, activacion relu
layer1 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
#segunda capa tiene 128 entradas(de capa 1) y 10 de salida, activacion softmax
layer2 = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)

#definir pesos y biases tomados del archivo
#transponer para evitar errores
layer1.weights = w1.T
layer1.bias = b1.T
#fabio dice que hacer transpuesta porque esto lo lee como 1000,784 cuando tiene que ser 784,1000
layer2.weights = w2.T
layer2.bias = b2.T

x = np.random.rand(1, 784)

output = layer1.forward(x)
salida = layer2.forward(output)

# %%
#predicciones
predict = np.argmax(salida, axis =1)
#acurracy
acurracy = np.mean(predict == x)

print("Salida capa 1:", output.shape)
print("Salida capa 2:", salida.shape)
print("Predicción:", acurracy)

# %%
#probar imagenes
#asegurar que tengan 784 entradas
c = images.reshape(-1, 784)
#normalizar
c = c/255

#forward de ambas capas con las imagenes
out = layer1.forward(c)
sal = layer2.forward(out)

# %%
#predicción
predict = np.argmax(sal, axis =1)
#acurracy
acurracy = np.mean(predict == labels)

print("Predicción:", acurracy * 100)

# %%



