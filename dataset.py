import torchvision
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms as T

# Rutas de los conjuntos de datos
PATH_TRAIN = "./processed"
PATH_VALID = "./validation"

DEVICE = 'cuda'

BATCH_SIZE = 16
LR = 0.001

# Transformaciones de datos de entrenamiento
train_augs = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation((0, 90)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformaciones de datos de validación
valid_augs = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Crear conjunto de datos de entrenamiento
trainset = torchvision.datasets.ImageFolder(root=PATH_TRAIN, transform=train_augs)

# Crear conjunto de datos de validación
valset = torchvision.datasets.ImageFolder(root=PATH_VALID, transform=valid_augs)

# Crear DataLoader para el conjunto de datos de entrenamiento
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

# Crear DataLoader para el conjunto de datos de validación
validloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
