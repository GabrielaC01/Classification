import torch 
from torch import nn 

class ASL_Model(nn.Module):
  def __init__(self, n_classes):
      super(ASL_Model, self).__init__()

      self.feature_extractor=nn.Sequential(
          nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5), padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=(4,4), stride=2),

          nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5,5), padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=(4,4), stride=2),

          nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5), padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=(4,4), stride=2),

          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), padding=1),
          nn.ReLU(),
 
      )
      self.maxpool=nn.MaxPool2d(kernel_size=(4,4), stride=2)

      self.classifier=nn.Sequential(
          nn.Flatten(),
          nn.Linear(6400,2048),
          nn.ReLU(),
          nn.Linear(2048, n_classes)
      )
      self.gradient=None

  def activations_hook(self, grad):
    self.gradient=grad

  def forward(self, images):
    x=self.feature_extractor(images) #activation_maps

    h=x.register_hook(self.activations_hook)
    x=self.maxpool(x)
    x=self.classifier(x)

    return x
  
  def get_activation_gradients(self): #a1, a2, a3, ... ak
    return self.gradient
  
  def get_activation(self, x): #A1, A2, A3, ... Ak
    return self.feature_extractor(x)