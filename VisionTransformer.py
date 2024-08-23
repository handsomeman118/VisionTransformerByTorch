from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
import torch
from torch import nn
from math import sqrt
import torch.nn.functional as F

# linear projection of flattend patches block
class LinearProjectionOfFlattendPatches(nn.Module):
  def __init__(self, patchWidth:int, patchHigh:int, c:int, w:int, h:int):
    super(LinearProjectionOfFlattendPatches, self).__init__()
    self.patchWidth = patchWidth
    self.patchHigh = patchHigh
    
    dim = self.patchHigh * self.patchWidth * c
    self.clsToken = nn.Parameter(torch.randn(1, dim))
    assert h % patchHigh == 0 and w % patchWidth == 0, 'h and w must be divided by ph and pw'
    self.positionEmbding = nn.Linear(dim, dim)
    
  def forward(self, x):
    x = rearrange(x, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph = self.patchHigh, pw = self.patchWidth)
    return  self.positionEmbding(torch.cat((repeat(self.clsToken, 'n d -> bs n d', bs = x.shape[0]), x), dim=1))
  

# transformerEncoder
  # MSA
class MSA(nn.Module):
  def __init__(self, k, N, D):
    super(MSA, self).__init__()
    self.k = k
    self.N = N
    self.D = D
    assert D % k == 0,f'D is {D}, k is {k}, D must be divided by k'
    self.Dh = D//k

    self.linearToQKV = nn.Linear(self.D, 3*self.D, bias=False)

  def forward(self, z:torch.Tensor):
    qkv = self.linearToQKV(z) # n,d -> n , k * 3Dh
    q, k, v = rearrange(qkv, 'b n (headNum m Dh) -> m b headNum n Dh', headNum = self.k, m = 3, Dh = self.Dh)
    A = F.softmax(q @ k.transpose(-2, -1) / sqrt(self.Dh))
    return rearrange((A @ v), 'b k n Dh -> b n (k Dh)', k = self.k, Dh = self.Dh) 

  # MSA block
class MSA_Block(nn.Module):
  def __init__(self, k, N, D):
    super(MSA_Block, self).__init__()
    self.MSA = MSA(k, N, D)
    self.Norm = nn.LayerNorm((D))

  def forward(self, z):
    return z + self.MSA(self.Norm(z))
  
  # MLP block
class MLP_block(nn.Module):
  def __init__(self, N:int, D:int):
    super(MLP_block, self).__init__()
    self.MLP = nn.Linear(D, D)
    self.Norm = nn.LayerNorm((D))
  
  def forward(self, z):
    return z + self.MLP(self.Norm(z))

  # TransformerEncoder block
class TransformerEncoder(nn.Module):
  def __init__(self, L:int, k:int, patchWidth:int, patchHigh:int, c:int, w:int, h:int):
    super(TransformerEncoder, self).__init__()
    self.TransformerEncoderBlock = nn.ModuleList()
    N = (h // patchHigh) * (w // patchWidth)
    D = (patchWidth * patchHigh * c)
    for _ in range(L):
      self.TransformerEncoderBlock.append(
        nn.Sequential(
          MSA_Block(k, N, D),
          MLP_block(N, D)
        ))
    
  def forward(self, z):
    for layer in self.TransformerEncoderBlock:
      z = layer(z)
    return z

# VIT
class VisionTransformer(nn.Module):
  def __init__(self, L: int, k: int, clsNum: int, c: int, w: int, h: int, patchWidth: int, patchHigh: int):
    '''
    ### L: int              ----------transformerEncoder层数\n
    ### k: int              ----------MSA的头数\n
    ### clsNum: int         ----------分类数\n
    ### bs: int             ----------batch_size的尺寸\n
    ### c: int              ----------数据的通道数 RGB\n
    ### w: int              ----------尺寸宽度   图片宽\n
    ### h: int              ----------尺寸高度   图片高\n
    ### patchWidth: int     ----------patch的宽度\n
    ### patchHigh: int      ----------patch的高度\n
    '''
    super(VisionTransformer, self).__init__()
    self.linearProjectionOfFlattendPatches = LinearProjectionOfFlattendPatches(patchWidth, patchHigh, c, w, h)
    self.transformerEncoder = TransformerEncoder(L, k, patchWidth, patchHigh, c, w, h)
    self.MLP_Head = nn.Linear(c * patchHigh * patchWidth, clsNum)
    
  def forward(self, x):
    token = self.linearProjectionOfFlattendPatches(x)
    z = self.transformerEncoder(token)
    y = self.MLP_Head(rearrange(z, 'b n d -> n b d')[0])
    return F.softmax(y)

if __name__ == '__main__':
  image = torch.randn(3, 3, 224, 224)
  net = VisionTransformer(1, 1, 10, 3, 224, 224, 16, 16)
  y = net(image)
  print(net, y.shape)
  # image = torch.randn(3, 3, 224, 224)
  # net = LinearProjectionOfFlattendPatches(3, 16, 16, 3, 224, 224)
  # y = net(image)
  # print(y.shape)