# coding:utf8
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

data = np.arange(32*9).reshape(9,32)    # 共九个句子 每个句子长度32
tensor_data = torch.from_numpy(data)
# print(data.size())


# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.embedding = nn.Embedding(32*9, 100,padding_idx=32*9-1)
#         # self.convs = nn.ModuleList(
#         #     [nn.Conv2d(1, 128, (k, 50)) for k in [3,5,7]])  # （ 输入层数，输出层数，卷机核大小，stride，padding）
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(1,128,(k,100),padding=(int((k-1)/2),0)) for k in [3,5,7]])
#         self.dropout = nn.Dropout(0.5)
#         self.fc = nn.Linear(4 * 3, 3)
#
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)).squeeze(3)    #[9, 32, 100]
#         print('-----',x.size())
#         # x = F.relu(conv(x)).squeeze(3).view(5,9,4)
#         # x = F.max_pool1d(x, x.size(2)).squeeze(2)    #[5, 50]
#         print('-----',x.size())
#         return x
#
#     def forward(self, x):
#         out = self.embedding(x)      # batchsize * pad_size * embed
#         print(out.shape)
#         out = out.unsqueeze(1)     # 此处1指的是在 x.size=（ ，，，）的第一个位置增加一个维度 batchsize * 1 * pad_size * embed
#
#         out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
#
#         out = self.dropout(out)
#         # out = self.fc(out)
#         return out
#
# model = Model()
# out = model.forward(tensor_data)      # [9,384,32]
# print(out.size())

##### 已经得到了cnn的输出 and then？

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(9*32, 100, padding_idx=9*32-1)
        self.gru = nn.GRU(100, 128, 2,bidirectional=True, batch_first=True, dropout=0.5)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(128 *2 ))

    def forward(self, x):
        # x, _ = x
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.gru(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]
        # 双向LSTM输出矩阵的的大小经过 两个相同大小的矩阵输出。
        M = self.tanh1(H)    #[9, 32, 256]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)   #[9, 32, 256]*[9, 32, 1]
        out = (H * alpha)  #  []    # [9, 32, 256]
        out = out.permute(0,2,1)    # [9,256,32]
        return out,emb
model = Model()
out,emb = model.forward(tensor_data)          # [9,256,32]


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
class MLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
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

'''part 3--CNN'''
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()

        # self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        # self.convs = nn.ModuleList(
        #     [nn.Conv2d(1, 128, (k, 100)) for k in [3,5,7]])  # （ 输入层数，输出层数，卷机核大小，stride，padding）
        self.dropout = nn.Dropout(0.5)
        # self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)


    def conv(self, x, k):
        conv = nn.Conv2d(1,128,(k,100),stride=1)
        x = F.relu(conv(x)).squeeze(3)
        print(x.size())
        # x = F.max_pool1d(x, x.size(2)).squeeze(2) 此处不应单独做卷积，而是等一个卷积核遍历完之后
        return x

    def forward(self, g_t, x):
        out = torch.cat([
              torch.cat([
              F.max_pool1d(
              torch.cat([self.conv(torch.cat([g_t[i].unsqueeze(0),x[i,j:j+k,:]],dim=0).unsqueeze(0).unsqueeze(0),k)
                        for j in range(32-k+1)],dim=2)
                        ,kernel_size=(32-k+1)*2).squeeze(2)
                        for k in [3,5,7]],dim=1)
                        for i in range(len(g_t))],dim=0)

        print(out.shape)
            # out = out.unsqueeze(1)     # 此处1指的是在 x.size=（ ，，，）的第一个位置增加一个维度 batchsize * 1 * pad_size * embed
            #
            # out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
            # out = self.dropout(out)
            # out = self.fc(out)
        return out


'''part 3--RCNN'''
class drnn(nn.Module):
    def __init__(self):
        super(drnn, self).__init__()
        # self.gru = nn.GRU(100, 128, 2, bidirectional=True, batch_first=True, dropout=0.5)
        # self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
        #                     bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(120)
        # self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)

    def gru(self,x):
        g = nn.GRU(100, 128, 2, bidirectional=True, batch_first=True, dropout=0.5)
        out,_ = g(x)
        return out

    def forward(self, g_t, x):
        # x, _ = x
        # embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        # out, _ = self.lstm(embed)
        out = torch.cat([
                    torch.cat([self.gru(
                        torch.cat([g_t[i].unsqueeze(0), x[i, j:j + 3, :]], dim=0).unsqueeze(0))
                               for j in range(32 - 3 + 1)], dim=1)
                        for i in range(len(g_t))], dim=0)

        # out = torch.cat((x, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)     # 调整维度信息，0，2，代表原来对应的维度，后交换
        out = self.maxpool(out).squeeze()
        # out = self.fc(out)
        return out
encoder2 = drnn()
out = encoder2.forward(g_t,emb)
print(out.size())