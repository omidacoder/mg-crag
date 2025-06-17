import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

random_seed = 52
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)

class ResidualBlockMain(nn.Module):
   def __init__(self, input_dim, output_dim):
      super(ResidualBlockMain, self).__init__()
      self.fc1 = nn.Linear(input_dim, output_dim)
      self.dropout1 = nn.Dropout(0.2)
      self.fc2 = nn.Linear(output_dim, output_dim)
      self.dropout2 = nn.Dropout(0.2)
      self.downsample = None
      if input_dim != output_dim:
        self.downsample = nn.Linear(input_dim, output_dim)

   def forward(self, x):
      residual = x
      x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
      x = self.dropout1(x)
      x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
      x = self.dropout2(x)
      if self.downsample is not None:
        residual = self.downsample(residual)
      x = x + residual
      return x

class ResNet(nn.Module):
   def __init__(self, input_len):
      super(ResNet, self).__init__()
      self.fc1 = nn.Linear(input_len, 2048)
      self.dropout1 = nn.Dropout(0.2)

      self.residual_block1 = ResidualBlockMain(2048, 2048)
      self.residual_block2 = ResidualBlockMain(2048, 1024)
      self.residual_block3 = ResidualBlockMain(1024, 512)
      self.residual_block4 = ResidualBlockMain(512, 256)
      self.residual_block5 = ResidualBlockMain(256, 128)
      self.residual_block6 = ResidualBlockMain(128, 64)
      self.residual_block7 = ResidualBlockMain(64, 32)
      self.residual_block8 = ResidualBlockMain(32, 16)
      self.residual_block9 = ResidualBlockMain(16, 8)

      self.fc10 = nn.Linear(8, 3)

   def forward(self, x):
      x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
      x = self.dropout1(x)

      x = self.residual_block1(x)
      x = self.residual_block2(x)
      x = self.residual_block3(x)
      x = self.residual_block4(x)
      x = self.residual_block5(x)
      x = self.residual_block6(x)
      x = self.residual_block7(x)
      x = self.residual_block8(x)
      x = self.residual_block9(x)

      output = self.fc10(x)
      return output