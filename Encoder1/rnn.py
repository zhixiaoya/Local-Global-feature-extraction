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
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = 300                                                # 字向量维度
        self.filter_sizes = [3,5,7]                                     # 卷积核尺寸
        self.hidden_size = 128                                          # 卷积核数量(channels数)
        self.num_layers = 2



class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab-1)
        self.gru = nn.GRU(config.embed, config.hidden_size, config.num_layers,bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(config.hidden_size *2 ))

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
# model = Model()
# out,emb = model.forward(tensor_data)          # [9,256,32]