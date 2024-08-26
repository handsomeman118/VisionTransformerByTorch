from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
import torch
from torch import nn
from math import sqrt
import torch.nn.functional as F

# linear projection of flattend patches block
class LinearProjectionOfFlattendPatches(nn.Module):
  def __init__(self, hiddenSize:int, patchWidth:int, patchHigh:int, c:int, w:int, h:int):
    assert h % patchHigh == 0 and w % patchWidth == 0, 'h and w must be divided by ph and pw'
    super(LinearProjectionOfFlattendPatches, self).__init__()
    
    self.clsToken = nn.Parameter(torch.randn(1, hiddenSize))
    self.flatten = nn.Conv2d(c, hiddenSize, (patchHigh, patchWidth), (patchHigh, patchWidth))
    self.positionEmbding = nn.Linear(hiddenSize, hiddenSize, bias=False)
  def forward(self, x):
    x = self.flatten(x)
    x = rearrange(x, 'b c h w -> b (h w) c')
    return  self.positionEmbding(torch.cat((repeat(self.clsToken, 'n d -> bs n d', bs = x.shape[0]), x), dim=1))
  

# transformerEncoder
  # MSA
class MSA(nn.Module):
  def __init__(self, heads:int, dim:int):
    assert dim % heads == 0,f'D is {dim}, k is {heads}, D must be divided by heads_num'
    super(MSA, self).__init__()
    self.heads = heads
    self.dim = dim
    self.Dh = dim // heads

    self.linearToQKV = nn.Linear(dim, 3*dim, bias=False)

  def forward(self, z:torch.Tensor):
    qkv = self.linearToQKV(z) # n,d -> n , k * 3Dh
    q, k, v = rearrange(qkv, 'b n (headNum m Dh) -> m b headNum n Dh', headNum = self.heads, m = 3, Dh = self.Dh)
    A = F.softmax(q @ k.transpose(-2, -1) / sqrt(self.Dh), dim=-1) # b headNum n n
    return rearrange((A @ v), 'b k n Dh -> b n (k Dh)', k = self.heads, Dh = self.Dh) 

  # MSA block
class MSA_Block(nn.Module):
  def __init__(self, heads:int, dim:int):
    super(MSA_Block, self).__init__()
    self.Norm = nn.LayerNorm((dim))
    self.MSA = MSA(heads, dim)
  def forward(self, z):
    return z + self.MSA(self.Norm(z))
  
  # MLP block
class MLP_block(nn.Module):
  def __init__(self, dim:int, MLP_dim:int):
    super(MLP_block, self).__init__()
    self.Norm = nn.LayerNorm((dim))
    self.MLP = nn.Sequential(
      nn.Linear(dim, MLP_dim),
      nn.GELU(),
      nn.Linear(MLP_dim, dim)
    )
  def forward(self, z):
    return z + self.MLP(self.Norm(z))

  # TransformerEncoder block
class TransformerEncoder(nn.Module):
  def __init__(self, layers:int, dim:int, MLP_dim:int, heads:int):
    super(TransformerEncoder, self).__init__()

    self.TransformerEncoderBlock = nn.ModuleList()
    for _ in range(layers):
      self.TransformerEncoderBlock.append(
        nn.Sequential(
          MSA_Block(heads, dim),
          MLP_block(dim, MLP_dim)
        ))
  def forward(self, z):
    for layer in self.TransformerEncoderBlock:
      z = layer(z)
    return z

# VIT
class VisionTransformer(nn.Module):
  def __init__(self, layers: int, dim: int, MLP_dim:int, heads: int, clsNum: int, c: int, h: int, w: int, patchHigh: int, patchWidth: int):
    '''
    ### layers: int              ----------transformerEncoder层数\n
    ### dim: int            ----------隐藏的映射维度\n
    ### MLP_dim: int        ----------MLP的隐藏层尺寸\n
    ### heads: int              ----------MSA的头数\n
    ### clsNum: int         ----------分类数\n
    ### c: int              ----------数据的通道数 RGB\n
    ### h: int              ----------尺寸高度   图片高\n
    ### w: int              ----------尺寸宽度   图片宽\n
    ### patchHigh: int      ----------patch的高度\n
    ### patchWidth: int     ----------patch的宽度\n
    '''
    super(VisionTransformer, self).__init__()

    self.linearProjectionOfFlattendPatches = LinearProjectionOfFlattendPatches(dim, patchWidth, patchHigh, c, w, h)
    self.transformerEncoder = TransformerEncoder(layers, dim, MLP_dim, heads)
    self.MLP_Head = nn.Linear(dim, clsNum)
  def forward(self, x):
    token = self.linearProjectionOfFlattendPatches(x)
    z = self.transformerEncoder(token)
    y = self.MLP_Head(rearrange(z, 'b n d -> n b d')[0])
    return F.softmax(y, dim=-1)

if __name__ == '__main__':
  image = torch.randn(3, 3, 96, 120)
  net = VisionTransformer(1, 768, 3072, 12, 10, 3, 120, 64, 40, 32)
  y = net(image)
  print(net, y.shape)