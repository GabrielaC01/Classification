from model import ASL_Model
from dataset import trainloader, trainset, validloader
import torch
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt 

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE=16
LR=0.001
EPOCHS=11



def train_fn(dataloader, model, optimizer, criterion):
  model.train()
  total_loss=0.0
  for images, labels in tqdm(dataloader):
    images=images.to(DEVICE)
    labels=labels.to(DEVICE)

    optimizer.zero_grad()
    logits= model(images)
    loss=criterion(logits, labels)
    loss.backward()
    optimizer.step()

    total_loss+=loss.item()
  return total_loss/len(dataloader)

def eval_fn(dataloader, model, criterion):
  model.eval()
  total_loss=0.0
  for images, labels in tqdm(dataloader):
    images=images.to(DEVICE) 
    labels=labels.to(DEVICE)

    logits= model(images)
    loss=criterion(logits, labels)

    total_loss+=loss.item()
  return total_loss/len(dataloader)
  
asl_model = ASL_Model(n_classes = len (trainset.class_to_idx)).to(DEVICE) 
#print(trainset.idx_to_class)
optimizer= torch.optim.Adam(asl_model.parameters(), lr=LR)
criterion= torch.nn.CrossEntropyLoss()

best_valid_loss=np.Inf

vector_train = []
vector_valid = []
vector_epoch = []

for i in range(EPOCHS):
  train_loss=train_fn(trainloader, asl_model, optimizer, criterion)
  valid_loss=eval_fn(validloader, asl_model, criterion)

  if valid_loss < best_valid_loss:
    #torch.save(asl_model.state_dict(), './weights/best_weigts.pt')
    torch.save(asl_model, f'./weights/best_weights_{EPOCHS}.pt')
    best_valid_loss=valid_loss
    print("SAVED_WEIGHTS_SUCCESS")
  
  print(f"EPOCH:{i+1} TRAIN LOSS: {train_loss} VALID LOSS: {valid_loss}")
  vector_epoch.append(i+1)
  vector_train.append(train_loss)
  vector_valid.append(valid_loss)
 
plt.plot(vector_epoch,vector_train, "o",  color ="red")
plt.plot(vector_epoch,vector_valid, "o", color="yellow")
plt.xlabel("EPOCHS", fontsize = 15, color = "blue")
plt.ylabel("LOSS", fontsize = 15, color = "blue")
plt.title("GRÁFICA DE LOSS", fontsize =18, color = "green")
plt.savefig(f'./image/graphics/grafica_{EPOCHS}.png')
plt.show()

 
