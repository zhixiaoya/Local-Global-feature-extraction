# coding=utf8
import torch.nn.functional as F
import torch
import torch.nn as nn

###### interaction mode:
###### same
def SAME(h):
    g_t = F.max_pool1d(h, h.size(2)).squeeze(2)
    # print(x.size())
    return g_t

###### attend

def ATTEND(h,x):

    avg_pool_x = F.avg_pool2d(x.unsqueeze(1),kernel_size =(3,100),stride=1,padding=(1,0)).squeeze(3)
    avg_pool_x = avg_pool_x.permute(0,2,1)
    alpha = F.softmax(torch.matmul(torch.tanh(h),avg_pool_x),dim=1)
    sum_alpha_h = torch.sum(h*alpha,dim=2)   # sum([9, 256, 32]*[9, 256, 1]=[9,256,32]) = [9, 256]
    maxpool_enc1 = F.max_pool1d(h, h.size(2)).squeeze(2) # [9, 256]
    g_t = torch.cat([maxpool_enc1,sum_alpha_h],dim=1)     #[9, 512]
    return g_t

g_t1 = SAME(out)
g_t2 = ATTEND(out,emb)
print(g_t1.size())
print(g_t2.size())

'''compress'''

def MLP(input_size, common_size):

    linear = nn.Sequential(
    nn.Linear(input_size, input_size // 2),
    nn.ReLU(inplace=True),
    nn.Linear(input_size // 2, input_size // 4),
    nn.ReLU(inplace=True),
    nn.Linear(input_size // 4, common_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

mlp = MLP(512,100)
g_t = mlp.forward(g_t2)
print(g_t.size())