from model import ASL_Model
from dataset import trainloader, trainset, validloader
import torch
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt 

# Verificar si está disponible la GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 16
LR = 0.001
EPOCHS = 1

# Función de entrenamiento
def train_fn(dataloader, model, optimizer, criterion):
  model.train()
  total_loss = 0.0
  for images, labels in tqdm(dataloader):
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
  return total_loss / len(dataloader)

# Función de evaluación
def eval_fn(dataloader, model, criterion):
  model.eval()
  total_loss = 0.0
  for images, labels in tqdm(dataloader):
    images = images.to(DEVICE) 
    labels = labels.to(DEVICE)

    logits = model(images)
    loss = criterion(logits, labels)

    total_loss += loss.item()
  return total_loss / len(dataloader)

# Crear una instancia del modelo ASL_Model
asl_model = ASL_Model(n_classes=len(trainset.class_to_idx)).to(DEVICE) 

# Optimizador y función de pérdida
optimizer = torch.optim.Adam(asl_model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

best_valid_loss = np.Inf

vector_train = []
vector_valid = []
vector_epoch = []

# Ciclo de entrenamiento
for i in range(EPOCHS):
  # Entrenamiento
  train_loss = train_fn(trainloader, asl_model, optimizer, criterion)
  
  # Evaluación en conjunto de validación
  valid_loss = eval_fn(validloader, asl_model, criterion)

  if valid_loss < best_valid_loss:
    # Guardar los mejores pesos del modelo
    torch.save(asl_model, f'./weights/best_weights_{EPOCHS}.pt')
    best_valid_loss = valid_loss
    print("SAVED_WEIGHTS_SUCCESS")
  
  print(f"EPOCH:{i+1} TRAIN LOSS: {train_loss} VALID LOSS: {valid_loss}")
  
  # Guardar las métricas de entrenamiento y validación para la gráfica
  vector_epoch.append(i+1)
  vector_train.append(train_loss)
  vector_valid.append(valid_loss)
 
# Graficar las métricas de pérdida
plt.plot(vector_epoch, vector_train, "o", color="red")
plt.plot(vector_epoch, vector_valid, "o", color="yellow")
plt.xlabel("EPOCHS", fontsize=15, color="blue")
plt.ylabel("LOSS", fontsize=15, color="blue")
plt.title("GRÁFICA DE LOSS", fontsize=18, color="green")
plt.savefig(f'./image/graphics/grafica_{EPOCHS}.png')
plt.show()
