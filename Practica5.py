# %%
import numpy as np
import DnnLib
import json
import argparse


# carga datos de npz para labels e imagenes
prueba = np.load("fashion_mnist_train.npz")
test = np.load("fashion_mnist_test.npz")

#separar en labels e imagenes
imagenes = prueba["images"]
labelE = prueba["labels"]

imagens = test["images"]
labelP = test["labels"]

#reformar las imagenes(entradas)
imagenE = imagenes.reshape(-1, 784).astype(np.float32) / 255
imagenP = imagens.reshape(-1, 784).astype(np.float32) / 255

# funcion de one hot
def to_one_hot(labels, num_classes=10):
    h = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    h[np.arange(labels.shape[0]), labels] = 1
    return h

#one hot para cada labels
yE = to_one_hot(labelE)
yP = to_one_hot(labelP)

# cargar json
def load_datos_reg():
    with open("mnist_entrenado.json", "r") as f:
        datos = json.load(f)

    # Capa 1 
    layer1 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
    
    # Dropout
    dropout1 = DnnLib.Dropout(dropout_rate=0.5)

    # Capa 2
    layer2 = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
    
    print("pesos")
    return [layer1, dropout1, layer2]

#funciones de forward y backward para droupout
def forward_pass_with_dropout(layers, x, training=True):
    activations = [x] 
    for layer in layers:
        if hasattr(layer, 'training'):
            layer.training = training
        activations.append(layer.forward(activations[-1]))
    return activations

def backward_pass_with_dropout(layers, grad_output):
    grad = grad_output
    for layer in reversed(layers):
        grad = layer.backward(grad)
    return grad

def entrenamiento_reg(capas, optimizers, x, y, label, X_val, y_val, epochs=50, batches=128):
    n = x.shape[0]
    for e in range(1, epochs+1):
        r = np.random.permutation(n)
        # print("Bajado correcto 1")
        x_shuffled = x[r]
        y_shuffled = y[r]
        # print("Bajado correcto 2")

        epoch_loss = 0.0
        n_batches, correct, total = 0, 0, 0

        for i in range(0, n, batches):
            x_batch = x_shuffled[i:i+batches]
            y_batch = y_shuffled[i:i+batches]
            # print("Bajado correcto 3")
            
            # forward y dropout 
            output = forward_pass_with_dropout(capas, x_batch, training=True)
            
            penultima = output[-2]
            out_linear = capas[-1].forward_linear(penultima)

            # perdida
            perdida = DnnLib.cross_entropy(out_linear, y_batch)

            # regularizar
            total_reg_loss =0.0
            total_reg_loss = capas[0].compute_regularization_loss() + capas[2].compute_regularization_loss()
            data_loss = perdida + total_reg_loss

            # backward con dropout
            gradiente = DnnLib.softmax_crossentropy_gradient(out_linear, y_batch)
            backward_pass_with_dropout(capas, gradiente)
            
            # Actualizar capas (no dropout)
            optimizers.update(capas[2])
            optimizers.update(capas[0])

            epoch_loss += perdida
            n_batches += 1
            preds = np.argmax(out_linear, axis=1)
            labels = np.argmax(y_batch, axis=1)
            correct += np.sum(preds == labels)
            total += len(labels)

        if e % 10 == 0:
            print(f"Data Loss: {perdida:.4f}, "
                f"Reg Loss: {total_reg_loss:.4f}, Total: {data_loss:.4f}")
            val_output = forward_pass_with_dropout(capas, X_val, training=False)
            ps = val_output[-2]
            out_linears = capas[-1].forward_linear(ps)
            val_loss = DnnLib.cross_entropy(out_linears, y_val)
            print(f"Epoch {e}, Train Loss: {perdida:.4f}, Val Loss: {val_loss:.4f}")

        avg_loss = epoch_loss / n_batches
        acc = correct / total
        print(f"Epoch {e}, Avg Loss: {avg_loss:.4f}, Acc: {acc*100:.2f}%")

# %%
def load_datos():
    with open("mnist_entrenado.json", "r") as f:
        datos = json.load(f)

    layer1 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)  
    dropout1 = DnnLib.Dropout(dropout_rate=0.5)
    layer2 = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
    
    layer1.weights = np.array(datos["layers"][0]["W"], dtype=np.float32)
    layer1.bias = np.array(datos["layers"][0]["b"], dtype=np.float32)

    layer2.weights = np.array(datos["layers"][1]["W"], dtype=np.float32)
    layer2.bias = np.array(datos["layers"][1]["b"], dtype=np.float32)

    print(layer1.weights.shape, layer1.bias.shape, layer2.weights.shape, layer2.bias.shape)

    return [layer1, dropout1, layer2]

    

# %%

imagenE = imagenes.reshape(-1, 784).astype(np.float32) / 255
imagenP = imagens.reshape(-1, 784).astype(np.float32) / 255

val_ratio = 0.1
n_train = imagenE.shape[0]
n_val = int(n_train * val_ratio)

indices = np.random.permutation(n_train)
train_idx = indices[n_val:]  # índice para entrenamiento
val_idx = indices[:n_val]    # índice para validación

X_train, y_train_labels = imagenE[train_idx], labelE[train_idx]
y_train = yE[train_idx]

X_val, y_val_labels = imagenE[val_idx], labelE[val_idx]
y_val = yE[val_idx]

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")


capas = load_datos()
for L in capas:
    if not hasattr(L, "training"):  
        L.set_regularizer(DnnLib.RegularizerType.L2, 1e-4)

optimizers = DnnLib.Adam(0.001)
entrenamiento_reg(capas, optimizers, imagenE, yE, labelE, X_val, y_val, epochs=10)

# %%

# carga datos de npz para labels e imagenes
prueba = np.load("fashion_mnist_train.npz")
test = np.load("fashion_mnist_test.npz")

#separar en labels e imagenes
imagenes = prueba["images"]
labelE = prueba["labels"]

imagens = test["images"]
labelP = test["labels"]

#reformar las imagenes(entradas)
imagenE = imagenes.reshape(-1, 784).astype(np.float32) / 255
imagenP = imagens.reshape(-1, 784).astype(np.float32) / 255
print(imagenE.shape)

#accuracy entre test y train de fashion mnist

capas = load_datos()

out = forward_pass_with_dropout(capas, imagenE)
out1 = forward_pass_with_dropout(capas, imagenP)

penultima = out[-2]
out_linear = capas[-1].forward_linear(penultima)

ps = out1[-2]
ol = capas[-1].forward_linear(ps)

# Para test
predicts = np.argmax(out_linear, axis=1)
acurr = np.mean(predicts == labelE)
print("Predicción entrenamiento:", acurr * 100)

# Para entrenamiento
predicts = np.argmax(ol, axis=1)
acurr = np.mean(predicts == labelP)   
print("Predicción test:", acurr * 100)

# %%

def get_args():
    parser = argparse.ArgumentParser(description="Entrenamiento con dropout y regularización para Fashion-MNIST- Valeria")

    parser.add_argument("--epochs", type=int, default=50,
                        help="Número de épocas para entrenar")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Tamaño de cada batch")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    imagenE = imagenes.reshape(-1, 784).astype(np.float32) / 255
    imagenP = imagens.reshape(-1, 784).astype(np.float32) / 255

    val_ratio = 0.1
    n_train = imagenE.shape[0]
    n_val = int(n_train * val_ratio)

    indices = np.random.permutation(n_train)
    train_idx = indices[n_val:]  # índice para entrenamiento
    val_idx = indices[:n_val]    # índice para validación

    X_train, y_train_labels = imagenE[train_idx], labelE[train_idx]
    y_train = yE[train_idx]

    X_val, y_val_labels = imagenE[val_idx], labelE[val_idx]
    y_val = yE[val_idx]

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")

    capas = load_datos_reg()
    for L in capas:
        if not hasattr(L, "training"):  # DenseLayer
            L.set_regularizer(DnnLib.RegularizerType.L2, 1e-4)

    optimizers = DnnLib.Adam(0.001)

    entrenamiento_reg(
        capas, optimizers,
        imagenE, yE, labelE,
        X_val, y_val,
        epochs=args.epochs,
        batches=args.batch_size
    )



