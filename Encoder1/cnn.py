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
        self.num_epochs = 20                                            # epoch��
        self.batch_size = 128                                           # mini-batch��С
        self.pad_size = 32                                              # ÿ�仰����ɵĳ���(�����)
        self.learning_rate = 1e-3                                       # ѧϰ��
        self.embed = 300                                                # ������ά��
        self.filter_sizes = [3,5,7]                                     # ����˳ߴ�
        self.num_filters = 128                                          # ���������(channels��)



class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, self.embed,padding_idx=config.n_vocab-1)
        # self.convs = nn.ModuleList(
        #     [nn.Conv2d(1, 128, (k, 50)) for k in [3,5,7]])  # �� ����������������������˴�С��stride��padding��
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
        out = out.unsqueeze(1)     # �˴�1ָ������ x.size=�� ���������ĵ�һ��λ������һ��ά�� batchsize * 1 * pad_size * embed

        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)

        out = self.dropout(out)
        # out = self.fc(out)
        return out

# model = Model()
# out = model.forward(tensor_data)      # [9,384,32]
# print(out.size())