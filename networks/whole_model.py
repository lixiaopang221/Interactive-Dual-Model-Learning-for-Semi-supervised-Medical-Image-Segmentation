import torch
import torch.nn as nn
import models.network as models


class MODULE(nn.Module):
   def __init__(self,):
      super().__init__()
      self.net_top=nn.Conv2d(1,3,1,1)
      self.M=models.DenseUnet_2d_dual()
   def forward(self, x):
      x= self.net_top(x)
      x =self.M(x)
      return x

class MODULE_single(nn.Module):
   def __init__(self,):
      super().__init__()
      self.net_top=nn.Conv2d(1,3,1,1)
      self.M=models.DenseUnet_2d()
   def forward(self, x):
      x= self.net_top(x)
      x =self.M(x)
      return x
