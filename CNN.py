import torch.nn as nn
import torch
import numpy as np
from utils import data_process,MyDataset
from torch.utils.data import DataLoader
from LSTM import train,test

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_sizes, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_sizes)
        # 这个嵌入层不需要训练
        self.constant_embedding = nn.Embedding(vocab_size, embed_sizes)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 最大时间汇聚层没有参数，因此可以共享此实例
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # 创建多个一维卷积层
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_sizes, c, k))

    def forward(self, inputs):
        # 沿着向量维度将两个嵌入层连结起来，
        # 每个嵌入层的输出形状都是（批量大小，词元数量，词元向量维度）连结起来
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # 根据一维卷积层的输入格式，重新排列张量，以便通道作为第2维
        embeddings = embeddings.permute(0, 2, 1)
        # 每个一维卷积层在最大时间汇聚层合并后，获得的张量形状是（批量大小，通道数，1）
        # 删除最后一个维度并沿通道维度连结
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs

def main():

    train_dir = './aclImdb_v1/aclImdb/train'  # 原训练集文件地址
    train_path = './aclImdb_v1/aclImdb/train.txt'  # 预处理后的训练集文件地址

    test_dir = './aclImdb_v1/aclImdb/test'  # 原训练集文件地址
    test_path = './aclImdb_v1/aclImdb/test.txt'  # 预处理后的训练集文件地址

    vocab = data_process(train_path, train_dir) # 数据预处理
    data_process(test_path, test_dir)
    np.save('vocab.npy', vocab) # 词典保存为本地
    vocab = np.load('vocab.npy', allow_pickle=True).item()  # 加载本地已经存储的vocab

    # 构建MyDataset实例
    train_data = MyDataset(text_path=train_path)
    test_data = MyDataset(text_path=test_path)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    # 生成模型
    embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
    net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

    train(model=net, train_data=train_loader, vocab=vocab, epoch=10,method='CNN')

    # 加载训练好的模型
    net.load_state_dict(torch.load('cnn.pkl', map_location=torch.device('cuda')))

    # 测试结果
    acc = test(model=net, test_data=test_loader, vocab=vocab)
    print(acc)

if __name__ == '__main__':
    main()