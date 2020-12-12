# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse     # 解析命令行参数和选项的标准模块
from sklearn.model_selection import train_test_split
from sklearn import svm

parser = argparse.ArgumentParser(description='Text Classification')
parser.add_argument('--encoder1',default='cnn',type=str,required=True, help='choose a model: cnn or rnn')
parser.add_argument('--interaction_mode', default='S', type=str, help='S or A')
parser.add_argument('--encoder2',default='cnn', type=str, help='cnn or drnn')
args = parser.parse_args()


if __name__ == '__main__':

    dataset = 'Dataset/Sogou.CS'  # 数据集

    encoder1 = args.encoder1
    interaction_mode = args.interaction_mode
    encoder2 = args.encoder2

    from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('Encoder1.' + encoder1)      # 导入模块
    config = x.Config(dataset)          # 配置模型参数

    np.random.seed(1)               # 没有使用GPU的时候设置的固定生成的随机数
    torch.manual_seed(1)            # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(1)   # 为所有的GPU设置种子
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab,X,Y = build_dataset(config)
    # _ = zip(X,Y)
    # np.random.shuffle(list(_))
    # X,Y = zip(*_)

    # 划分数据集
    x_train,x_text,y_train,y_text = train_test_split(X,Y,test_size=0.3,shuffle=True)
    x_dev,x_text,y_dev,y_text = train_test_split(x_text,y_text,test_size=0.3,shuffle=True)

    # 创建迭代器
    train_iter = build_iterator(x_train,y_train, config)
    dev_iter = build_iterator(x_dev,y_dev, config)
    test_iter = build_iterator(x_text,y_text, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)                   # 词表长度
    model = x.Model(config).to(config.device)     # cuda/cpu
    # if model_name != 'Transformer':               # 除transformer,其他模型都要权重初始化
    #     init_network(model)
    print(model.parameters)                       # 打印模型参数
    # (模型参数，模型，训练集，验证集，测试集)
    train(config, model, train_iter, test_iter)


