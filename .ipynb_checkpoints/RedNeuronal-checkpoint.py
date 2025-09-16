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
prueba = np.load("mnist_train.npz")

#imagenes y labels de test
images = data["images"]
labels = data["labels"]

#imagenes y labels de train
image = prueba["images"]
label = prueba["labels"]

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
#probar con el mnist test
#asegurar que tengan 784 entradas
c = images.reshape(-1, 784)
#normalizar
c = c/255

f = image.reshape(-1,784)
f = f/255

#forward de ambas capas con las imagenes
out = layer1.forward(c)
sal = layer2.forward(out)

#forward de ambas capas con train
sali = layer1.forward(f)
outs = layer2.forward(sali)

# %%
#predicción con imagenes, validar que accurate tiene los labels
predict = np.argmax(sal, axis =1)
#acurracy
acurracy = np.mean(predict == labels)

print("Predicción test:", acurracy * 100)

#prediccion de train
predict = np.argmax(outs, axis =1)
#acurracy
acurracy = np.mean(predict == label)

print("Predicción entrenamiento:", acurracy * 100)



# %%
#entrenamiento de red neuronal
#creando arreglos & variables necesarias

#arreglo de layers
layers =[layer1, layer2]

#arreglo de optimizadores
optimizers = [
    # ("SGD", DnnLib.SGD(0.001)),
    # ("SGD+Momentum", DnnLib.SGD(0.001, 0.9)),
    ("Adam", DnnLib.Adam(0.001))
    # ("RMSprop", DnnLib.RMSprop(0.001))
]

epochs = 100

y = np.zeros((60000, 10), dtype=np.float64)
y[np.arange(60000), label] = 1.0


# %%
#funcion de entrenamiento
#arreglo de layers ya definidio, arreglo de todos los optimizadores, f son las entradas del mnist-train, y es el one hot, label a lo que estoy comparando y epochs epocas predefinidas 
def entrenamiento(optimizers, f, y, label, epochs):
    loss = []

    for opt_name, optimizer in optimizers:
        print(f"\n--- Training with {opt_name} ---")
        # Reset network weights (create new layers)
        layers = [ layer1, layer2 ]
        optimizer.reset()
        
        for e in range(epochs):
            #forward de ambas capas
            output = layers[0].forward(f)
            salida = layers[1].forward(output)
    
            #funcion de perdida
            #cross entropy es mas recomendada en mnist
            perdida = DnnLib.cross_entropy(salida, y)
    
            #gradiente de funcion de perdida
            gradiente = DnnLib.cross_entropy_gradient(salida, y)
    
            #al tener solo dos capas se puede hacer el backpropagation directamente
            #si no se debe utilizar for
            #empezar por ultima capa a la primera
            gradiente = layers[1].backward(gradiente)
            gradiente = layers[0].backward(gradiente)

            #updeater el optimizador
            optimizer.update(layers[1])
            optimizer.update(layers[0])

            if e % 20 == 0:
                predict = np.argmax(salida, axis =1)
                acurracy = np.mean(predict == label)
                loss.append(perdida)
                print(f"Epoch {e}, Loss: {perdida:.4f}, Accuracy: {acurracy:.4f}")
    return loss

# %%
per = []
per = entrenamiento(optimizers, f, y, label, epochs)
print(per)

# %%



