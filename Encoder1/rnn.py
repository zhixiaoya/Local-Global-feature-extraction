import torch
import torch.nn as nn
import torch.nn.functional as F

class Config(object):

    """���ò���"""
    def __init__(self, dataset, embedding):
        self.model_name = 'CNN'
        self.data_path = dataset + "/Data"

        self.vocab_path = dataset + '/vocab.pkl'  # �ʱ�
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # ģ��ѵ�����
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # �豸
        self.dataSeg_save = dataset + '/data_cutword_list.txt'
        self.label_save_path = dataset + '/label_save_path.txt'
        self.stopwords_path = dataset + '/hit_stopwords.txt'
        self.data_articles_vector = dataset + '/data_articles_vector.txt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # �豸

        self.dropout = 0.5                                              # ���ʧ��
        self.require_improvement = 1000                                 # ������1000batchЧ����û����������ǰ����ѵ��
        # self.num_classes = len(self.class_list)                         # �����
        self.n_vocab = 0                                                # �ʱ��С��������ʱ��ֵ
        self.num_epochs = 10                                            # epoch��
        self.batch_size = 128                                           # mini-batch��С
        self.pad_size = 32                                              # ÿ�仰����ɵĳ���(�����)
        self.learning_rate = 1e-3                                       # ѧϰ��
        self.embed = 300                                                # ������ά��
        self.filter_sizes = [3,5,7]                                     # ����˳ߴ�
        self.hidden_size = 128                                          # ���������(channels��)
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
        # ˫��LSTM�������ĵĴ�С���� ������ͬ��С�ľ��������
        M = self.tanh1(H)    #[9, 32, 256]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)   #[9, 32, 256]*[9, 32, 1]
        out = (H * alpha)  #  []    # [9, 32, 256]
        out = out.permute(0,2,1)    # [9,256,32]
        return out,emb
# model = Model()
# out,emb = model.forward(tensor_data)          # [9,256,32]