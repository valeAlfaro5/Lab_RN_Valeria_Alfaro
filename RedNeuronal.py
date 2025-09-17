# %%
import numpy as np
import DnnLib
import json
import random
import matplotlib.pyplot as plt

# %%
#carga de datos
with open("mnist_mlp_pretty.json", "r") as f:
    datos = json.load(f)

#carga imagenes y labels
data = np.load("mnist_test.npz")
images = data["images"]
labels = data["labels"]

pruebas = np.load("mnist_train.npz")
image = pruebas["images"]
label = pruebas["labels"]

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
#definir capas densas (ambas son densas segun el json) - parte 3
#primera capa tiene 784 entradas y 128 salidas, activacion relu
layer1 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
#segunda capa tiene 128 entradas(de capa 1) y 10 de salida, activacion softmax
layer2 = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)

#definir pesos y biases tomados del archivo
#transponer para evitar errores
layer1.weights = w1.T
layer1.bias = b1.T
layer2.weights = w2.T
layer2.bias = b2.T

x = np.random.rand(1, 784)

output = layer1.forward(x)
salida = layer2.forward(output)

# %%
#predicciones de prueba
predict = np.argmax(salida, axis =1)
#acurracy
acurracy = np.mean(predict == x)

#verificar que funcione la salidas de cada una con su predicción
print("Salida capa 1:", output.shape)
print("Salida capa 2:", salida.shape)
print("Predicción:", acurracy)

# %%
#probar con el mnist train
#asegurar que tengan 784 entradas
c = images.reshape(-1, 784)
#normalizar
c = c/255

#asegurar que tengan 784 entradas
f = image.reshape(-1, 784) / 255

#forward de ambas capas con las imagenes
out = layer1.forward(c)
sal = layer2.forward(out)

#forward con entrenamiento
salidas = layer1.forward(f)
outs = layer2.forward(salidas)

# %%
# Para test
predicts = np.argmax(sal, axis=1)
acurr = np.mean(predicts == labels)   # (test)
print("Predicción test:", acurr * 100)

# Para entrenamiento
predicts = np.argmax(outs, axis=1)
acurr = np.mean(predicts == label)    #   (train)
print("Predicción entrenamiento:", acurr * 100)


# %%
#arreglo de layers ya definidio, arreglo de todos los optimizadores, f son las entradas del mnist-train, y es el one hot, label a lo que estoy comparando y epochs epocas predefinidas, batch sizes para que pueda correr sin tronar 
def entrenamiento1(capas, optimizer, f, y_onehot, label, epochs=50, batch_size=128):
    n = f.shape[0]
    for e in range(epochs):
        # barajar los datos
        indices = np.random.permutation(n)
        f, y_onehot, label = f[indices], y_onehot[indices], label[indices]

        # mini-batches
        for i in range(0, n, batch_size):
            X_batch = f[i:i+batch_size]
            y_batch = y_onehot[i:i+batch_size]
            label_batch = label[i:i+batch_size]

            # forward
            out1 = capas[0].forward(X_batch)
            out2 = capas[1].forward(out1)

            # backward y update
            grad = DnnLib.cross_entropy_gradient(out2, y_batch)
            grad = capas[1].backward(grad)
            grad = capas[0].backward(grad)

            optimizer.update(capas[1])
            optimizer.update(capas[0])

        # imprimir cada 10 épocas
        if e % 10 == 0:
            preds = np.argmax(out2, axis=1)
            acc = np.mean(preds == label_batch)
            print(f"Epoch {e}, Loss: {DnnLib.cross_entropy(out2, y_batch):.4f}, Acc: {acc:.4f}")


# %%
#entrenamiento de red neuronal - parte 4
#creando arreglos & variables necesarias

capa1 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
#segunda capa tiene 128 entradas(de capa 1) y 10 de salida, activacion softmax
capa2 = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)

capa1.weights = np.random.randn(128,784)*0.01
capa1.bias = np.zeros(128,)
capa2.weights = np.random.randn(10, 128)*0.01
capa2.bias = np.zeros(10,)

capa = [capa1, capa2]

#arreglo de optimizadores
optimizers = [
    ("Adam", DnnLib.Adam(0.001))
]

epochs = 100

#creación de one hot
n = label.shape[0]  
y = np.zeros((n, 10), dtype=np.float64)
y[np.arange(n), label] = 1.0



# %%
#llamada de función de entrenamiento
per = []
per = entrenamiento1(capa, optimizers[0][1], f, y, label, epochs)
print(per)

# %%



