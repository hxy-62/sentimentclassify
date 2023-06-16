import torch # torch==1.7.1
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import re
import numpy as np
from tqdm import tqdm
from utils import tokenize,clean_str,data_process,MAX_LEN,text_transform,MyDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size) # embedding层

        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=False)
        self.decoder = nn.Linear(num_hiddens, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        # inputs的形状是（批量大小，词数），因此LSTM需要将序列长度（Seq_len）作为第一维，所以将输入转置后 再提取词特征
        embeddings = self.embedding(inputs.permute(1,0)) # permute(1,0)交换维度
        # LSTM只传入输入embeddings,因此只返回最后一层的隐藏层再各时间步的隐藏状态
        # outputs的形状是（词数，批量大小， 隐藏单元个数）
        outputs, _ = self.encoder(embeddings)
        # 连接初时间步和最终时间步的隐藏状态作为全连接层的输入。形状为(批量大小， 隐藏单元个数)
        encoding = outputs[-1] # 取LSTM最后一层结果
        outs = self.softmax(self.decoder(encoding)) # 输出层为二维概率[a,b]
        return outs

# 模型训练
def train(model, train_data, vocab, epoch=10,method='LSTM'):
    print('train model')
    model = model.to(device)
    loss_sigma = 0.0
    correct = 0.0
    # 定义损失函数和优化器
    if method == 'LSTM':
        criterion = torch.nn.NLLLoss()
    elif method == 'CNN':
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    for epoch in tqdm(range(epoch)):
        model.train()
        avg_loss = 0  # 平均损失
        avg_acc = 0  # 平均准确率
        for idx, (text, label) in enumerate(tqdm(train_data)):

            train_x = text_transform(text, vocab).to(device)
            train_y = label.to(device)

            optimizer.zero_grad()
            pred = model(train_x)
            if method == 'LSTM':
                pred = pred.log()
            loss = criterion(pred, train_y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            avg_acc += accuracy(pred, train_y)
        # 一个epoch结束后，计算平均loss和评平均acc
        avg_loss = avg_loss / len(train_data)
        avg_acc = avg_acc / len(train_data)

        print("avg_loss:", avg_loss, " train_avg_acc:,", avg_acc)

        # 保存训练完成后的模型参数
        if method == 'LSTM':
            torch.save(model.state_dict(), 'LSTM_IMDB_parameter.pkl')
        elif method == 'CNN':
            torch.save(model.state_dict(), 'CNN.pkl')


# 模型测试
def test(model, test_data, vocab):
    print('test model')
    model = model.to(device)
    model.eval()
    avg_acc = 0
    for idx, (text, label) in enumerate(tqdm(test_data)):
        train_x = text_transform(text, vocab).to(device)
        train_y = label.to(device)
        pred = model(train_x)
        avg_acc += accuracy(pred, train_y)
    avg_acc = avg_acc / len(test_data)
    return avg_acc

# 计算预测准确性
def accuracy(y_pred, y_true):
    label_pred = y_pred.max(dim=1)[1]
    acc = len(y_pred) - torch.sum(torch.abs(label_pred-y_true)) # 正确的个数
    return acc.detach().cpu().numpy() / len(y_pred)

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
    model = LSTM(vocab=vocab, embed_size=300, num_hiddens=128, num_layers=2)  # 定义模型

    train(model=model, train_data=train_loader, vocab=vocab, epoch=10,method="LSTM")

    # 加载训练好的模型
    model.load_state_dict(torch.load('LSTM_IMDB_parameter.pkl', map_location=torch.device('cuda')))

    # 测试结果
    acc = test(model=model, test_data=test_loader, vocab=vocab)
    print(acc)

if __name__ == '__main__':
    main()

