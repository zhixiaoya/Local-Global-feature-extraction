import torch
import torch.nn as nn
import torch.nn.functional as F

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'CNN'
        self.data_path = dataset + "/Data"

        self.vocab_path = dataset + '/vocab.pkl'  # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.dataSeg_save = dataset + '/data_cutword_list.txt'
        self.label_save_path = dataset + '/label_save_path.txt'
        self.stopwords_path = dataset + '/hit_stopwords.txt'
        self.data_articles_vector = dataset + '/data_articles_vector.txt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        # self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = 300                                                # 字向量维度
        self.filter_sizes = [3,5,7]                                     # 卷积核尺寸
        self.num_filters = 128                                          # 卷积核数量(channels数)



class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, self.embed,padding_idx=config.n_vocab-1)
        # self.convs = nn.ModuleList(
        #     [nn.Conv2d(1, 128, (k, 50)) for k in [3,5,7]])  # （ 输入层数，输出层数，卷机核大小，stride，padding）
        self.convs = nn.ModuleList(
            [nn.Conv2d(1,config.num_filters,(k,config.embed),padding=(int((k-1)/2),0)) for k in [3,5,7]])
        self.dropout = nn.Dropout(config.dropout)
        # self.fc = nn.Linear(4 * 3, 3)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)    #[9, 32, 100]
        print('-----',x.size())
        # x = F.relu(conv(x)).squeeze(3).view(5,9,4)
        # x = F.max_pool1d(x, x.size(2)).squeeze(2)    #[5, 50]
        print('-----',x.size())
        return x

    def forward(self, x):
        out = self.embedding(x)      # batchsize * pad_size * embed
        print(out.shape)
        out = out.unsqueeze(1)     # 此处1指的是在 x.size=（ ，，，）的第一个位置增加一个维度 batchsize * 1 * pad_size * embed

        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)

        out = self.dropout(out)
        # out = self.fc(out)
        return out

# model = Model()
# out = model.forward(tensor_data)      # [9,384,32]
# print(out.size())