from model import Model
from dataset import trainloader, trainset, validloader
import torch
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt 

# Verificar si está disponible la GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 16
LR = 0.001
EPOCHS = 50

# Función de entrenamiento
def train_fn(dataloader, model, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0
    for images, labels in tqdm(dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calcular las predicciones y el accuracy
        _, predicted_labels = torch.max(logits, 1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_examples += labels.size(0)

    accuracy = correct_predictions / total_examples * 100
    return total_loss / len(dataloader), accuracy

# Función de evaluación
def eval_fn(dataloader, model, criterion):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0
    for images, labels in tqdm(dataloader):
        images = images.to(DEVICE) 
        labels = labels.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()

        # Calcular las predicciones y el accuracy
        _, predicted_labels = torch.max(logits, 1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_examples += labels.size(0)

    accuracy = correct_predictions / total_examples * 100
    return total_loss / len(dataloader), accuracy

# Crear una instancia del modelo Model
model = Model(n_classes=len(trainset.class_to_idx)).to(DEVICE) 

# Optimizador y función de pérdida
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

best_valid_loss = np.Inf

vector_train_loss = []
vector_train_accuracy = []
vector_valid_loss = []
vector_valid_accuracy = []
vector_epoch = []

# Ciclo de entrenamiento
for i in range(EPOCHS):
    # Entrenamiento
    train_loss, train_accuracy = train_fn(trainloader, model, optimizer, criterion)
  
    # Evaluación en conjunto de validación
    valid_loss, valid_accuracy = eval_fn(validloader, model, criterion)

    if valid_loss < best_valid_loss:
        # Guardar los mejores pesos del modelo
        torch.save(model, f'./weights/best_weights_{EPOCHS}.pt')
        best_valid_loss = valid_loss
        print("SAVED_WEIGHTS_SUCCESS")
  
    print(f"EPOCH:{i+1} TRAIN LOSS: {train_loss} VALID LOSS: {valid_loss}")
    print(f"TRAIN ACCURACY: {train_accuracy}% VALID ACCURACY: {valid_accuracy}%")
  
    # Guardar las métricas de entrenamiento y validación para la gráfica
    vector_epoch.append(i+1)
    vector_train_loss.append(train_loss)
    vector_train_accuracy.append(train_accuracy)
    vector_valid_loss.append(valid_loss)
    vector_valid_accuracy.append(valid_accuracy)
 
# Graficar las métricas de pérdida y precisión
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(vector_epoch, vector_train_loss, "o", color="red", label="Train Loss")
plt.plot(vector_epoch, vector_valid_loss, "o", color="black", label="Valid Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(vector_epoch, vector_train_accuracy, "o", color="blue", label="Train Accuracy")
plt.plot(vector_epoch, vector_valid_accuracy, "o", color="green", label="Valid Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(f'./image/graphics/metrics_{EPOCHS}.png')
plt.show()