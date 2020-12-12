import torch.nn as nn
import torch

import torch.nn.functional as F

'''MAXPOOL1D'''
print('----MAXPOOL1D begin----')
m = nn.MaxPool1d(kernel_size=50)
input = torch.randn(20,16,50)
output = m(input).squeeze()
print(output.size())
print('----MAXPOOL1D end----')


'''MAXPOOL2D'''
print('----MAXPOOL2D begin----')
m = nn.MaxPool2d(kernel_size=32)
input = torch.randn(20,16,50,32)
output = m(input)
print(output.size())
print('----MAXPOOL2D end----')

'''AVGPOOL1D'''
print('----AVGPOOL1D begin----')
m = nn.AvgPool1d(kernel_size=7)
output = m(torch.tensor([[[1,2,3,4,5,6,7]]]))
print(output.size())
print('----AVGPOOL1D end----')

'''AVGPOOL2D'''
print('----AVGPOOL2D begin----')
m = nn.AvgPool2d(kernel_size=(50,32))
input = torch.randn(20,16,50,32)
output = m(input)
print(output.size())
print('----AVGPOOL2D end----')

'''F.avg_pool1d'''
print('----F.avg_pool1d begin----')
input = torch.tensor([[[1,2,3,4,5,6,7]]],dtype=torch.float32)

output = F.avg_pool1d(input,kernel_size=7)
print(output.size())
print('----avg_pool1d end----')

'''F.avg_pool2d'''
print('----F.avg_pool2d begin----')
input = torch.randn(20,16,50,32)
output = F.avg_pool2d(input,kernel_size=32)
print(output.size())
print('----avg_pool2d end----')