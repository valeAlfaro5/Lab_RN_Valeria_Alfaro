# %%
import numpy as np
import DnnLib
import json

# carga datos de npz para labels e imagenes
prueba = np.load("fashion_mnist_train.npz")
test = np.load("fashion_mnist_test.npz")

#separar en labels e imagenes
imagenes = prueba["images"]
labelE = prueba["labels"]

imagens = test["images"]
labelP = test["labels"]

print("IMAGE Train shape:", imagenE.shape, "Test shape:", imagenP.shape)
print("LABEL Train shape:", imagenE.shape, "Test shape:", imagenP.shape)

# funcion de one hot
def to_one_hot(labels, num_classes=10):
    h = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    h[np.arange(labels.shape[0]), labels] = 1
    return h

#one hot para cada labels
yE = to_one_hot(labelE)
yP = to_one_hot(labelP)

# cargar json
def load_datos():
    with open("mnist_entrenado.json", "r") as f:
        datos = json.load(f)

    # Capa 1 y regularizador
    layer1 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
    layer1.set_regularizer(DnnLib.RegularizerType.L2, 0.001)

    # Dropout
    dropout1 = DnnLib.Dropout(dropout_rate=0.5)

    # Capa 2 y regularizador
    layer2 = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
    layer2.set_regularizer(DnnLib.RegularizerType.L2, 0.001)

    # layer1.weights = np.array(datos["layers"][0]["W"], dtype=np.float32) # (128,784)
    # layer1.bias = np.array(datos["layers"][0]["b"], dtype=np.float32)  # (128,)

    # layer2.weights = np.array(datos["layers"][1]["W"], dtype=np.float32)  # (10,128)
    # layer2.bias = np.array(datos["layers"][1]["b"], dtype=np.float32)  # (10,)

    return [layer1, dropout1, layer2]

#funciones de forward y backward para droupout
def forward_pass_with_dropout(layers, x, training=True):
    activation = x
    for layer in layers:
        if isinstance(layer, DnnLib.Dropout):
            layer.training = training
        activation = layer.forward(activation)
    return activation

def backward_pass_with_dropout(layers, grad_output):
    grad = grad_output
    for layer in reversed(layers):
        grad = layer.backward(grad)
    return grad

def entrenamiento(capas, optimizers, x, y, label, epochs=50, batches=128):
    n = x.shape[0]
    for e in range(epochs):
        r = np.random.permutation(n)
        # print("Bajado correcto 1")
        x_shuffled = x[r]
        y_shuffled = y[r]
        labels = label[r]
        # print("Bajado correcto 2")

        epoch_loss = 0.0
        n_batches, correct, total = 0, 0, 0

        for i in range(0, n, batches):
            x_batch = x_shuffled[i:i+batches]
            y_batch = y_shuffled[i:i+batches]
            label_batch = labels[i:i+batches]
            # print("Bajado correcto 3")

            # forward y dropout 
            output = forward_pass_with_dropout(capas, x_batch, training=True)

            # perdida
            perdida = DnnLib.cross_entropy(output, y_batch)

            # regularizar
            # total_reg_loss = capas[0].compute_regularization_loss() + capas[2].compute_regularization_loss()
            # data_loss = perdida + total_reg_loss

            # backward con dropout
            gradiente = DnnLib.softmax_crossentropy_gradient(output, y_batch)
            gradiente = backward_pass_with_dropout(capas, gradiente)

            # Actualizar capas (no dropout)
            optimizers.update(capas[2])
            optimizers.update(capas[0])

            epoch_loss += perdida
            n_batches += 1
            preds = np.argmax(output, axis=1)
            correct += np.sum(preds == label_batch)
            total += len(label_batch)

            # if e % 2 == 0:
                # val_output = forward_pass_with_dropout(capas, imt, training=False)
                # val_loss = DnnLib.cross_entropy(val_output, labelt)
                # print(f"Epoch {e}, Train Loss: {perdida:.4f}")
            #     print(f"Reg Loss: {total_reg_loss:.4f}, Total: {data_loss:.4f}")

        avg_loss = epoch_loss / n_batches
        acc = correct / total
        print(f"Epoch {e}, Avg Loss: {avg_loss:.4f}, Acc: {acc*100:.2f}%")

# %%
# la = labelE.shape[0]
# y = np.zeros((la, 10), dtype=np.float64)
# y[np.arange(la), labelE] = 1.0


#reformar las imagenes(entradas)
imagenE = imagenes.reshape(-1, 784).astype(np.float32) / 255
imagenP = imagens.reshape(-1, 784).astype(np.float32) / 255


capas = load_datos()
optimizers = DnnLib.Adam(0.001)
entrenamiento(capas, optimizers, imagenE, yE, labelE, epochs=10)


# %%
def entrenar_cero():

    layer1 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
    layer2 = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)

    return [layer1, layer2]

# %%
import numpy as np
import DnnLib
import json

prueba = np.load("fashion_mnist_train.npz")
test = np.load("fashion_mnist_test.npz")

imagenes = prueba["images"]
labelE = prueba["labels"]

imagens = test["images"]
labelP = test["labels"]

imagenE = imagenes.reshape(-1, 784).astype(np.float32)/255
imagenP = imagens.reshape(-1, 784).astype(np.float32)/255

print("Train shape:", imagenE.shape, "Test shape:", imagenP.shape)

def to_one_hot(labels, num_classes=10):
    h = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    h[np.arange(labels.shape[0]), labels] = 1
    return h

yE = to_one_hot(labelE)
yP = to_one_hot(labelP)

def load_datos():
    with open("mnist_entrenado.json", "r") as f:
        datos = json.load(f)

    # Capa 1
    layer1 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
    layer1.set_regularizer(DnnLib.RegularizerType.L2, 0.001)

    # Dropout
    dropout1 = DnnLib.Dropout(dropout_rate=0.5)

    # Capa 2
    layer2 = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
    layer2.set_regularizer(DnnLib.RegularizerType.L2, 0.001)

    layer1.weights = np.array(datos["layers"][0]["W"], dtype=np.float32)  # (128,784)
    layer1.bias = np.array(datos["layers"][0]["b"], dtype=np.float32)  # (128,)

    layer2.weights = np.array(datos["layers"][1]["W"], dtype=np.float32)  # (10,128)
    layer2.bias = np.array(datos["layers"][1]["b"], dtype=np.float32)  # (10,)

    return [layer1, dropout1, layer2]

def forward_pass_with_dropout(layers, x, training=True):
    activation = x
    for layer in layers:
        if isinstance(layer, DnnLib.Dropout):
            layer.training = training
        activation = layer.forward(activation)
    return activation

def backward_pass_with_dropout(layers, grad_output):
    grad = grad_output
    for layer in reversed(layers):
        grad = layer.backward(grad)
    return grad


def entrenamiento(capas, optimizers, x, y, label, epochs=50, batches=128):
    n = x.shape[0]
    for e in range(epochs):
        r = np.random.permutation(n)
        x_shuffled = x[r]
        y_shuffled = y[r]
        labels = label[r]

        epoch_loss = 0.0
        n_batches, correct, total = 0, 0, 0

        for i in range(0, n, batches):
            x_batch = x_shuffled[i:i+batches]
            y_batch = y_shuffled[i:i+batches]
            label_batch = labels[i:i+batches]

            # Forward con dropout activo
            output = forward_pass_with_dropout(capas, x_batch, training=True)

            # perdida
            perdida = DnnLib.cross_entropy(output, y_batch)

            # Regularización
            total_reg_loss = capas[0].compute_regularization_loss() + capas[2].compute_regularization_loss()
            data_loss = perdida + total_reg_loss

            # gradiente y backward
            gradiente = DnnLib.softmax_crossentropy_gradient(output, y_batch)
            gradiente = backward_pass_with_dropout(capas, gradiente)

            # actualizar las capas menos el dropout
            optimizers.update(capas[0])
            optimizers.update(capas[2])

            epoch_loss += perdida
            n_batches += 1
            preds = np.argmax(output, axis=1)
            correct += np.sum(preds == label_batch)
            total += len(label_batch)
 
            if e % 2 == 0 and i == 0:
                print(f"Reg Loss: {total_reg_loss:.4f}, Total: {data_loss:.4f}")

        #AVERAGE LOSS Y ACCURACY
        avg_loss = epoch_loss / n_batches
        acc = correct / total
        if e % 5 == 0:
            print(f"Epoch {e}, Avg Loss: {avg_loss:.4f}, Acc: {acc*100:.2f}%")// mi fucking kernel muere

# %%
cap = load_datos()
print(cap)


