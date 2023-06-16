import os
import operator
from utils import clean_str,tokenize,get_common
from nltk.corpus import stopwords
import string
import numpy as np

vocab = get_common()
list1 = [0]*89527
dict1 = dict(zip(vocab,list1))

def get_dict(file_path,dict):
    for filename in os.listdir(file_path):
        if filename.endswith('.txt'):
            with open(os.path.join(file_path, filename), 'r',encoding='utf-8') as file:
                sentence = clean_str(file.readline())
                words = tokenize(sentence)
                #print(words)
                for i in words:
                    if i in vocab:
                        dict[i] += 1
                    else:
                        continue

get_dict('./aclImdb_v1/aclImdb/train/pos',dict1)
get_dict('./aclImdb_v1/aclImdb/train/neg',dict1)

list2 = []
#获取频率最高的2000个单词
for k,v in sorted(dict1.items(), key=operator.itemgetter(1),reverse=True)[:2000]: 
    list2.append(k)
stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours', 
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their', 
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once', 
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you', 
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will', 
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be', 
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself', 
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both', 
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn', 
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about', 
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn', 
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']

feature_words = [w for w in list2 if w not in stoplist]
#print(feature_words,len(feature_words))

documents = []

def get_document(file_path):
    for filename in os.listdir(file_path):
        if filename.endswith('.txt'):
            with open(os.path.join(file_path, filename), 'r',encoding='utf-8') as file:
                sentence = clean_str(file.readline())
                words = tokenize(sentence)
                if file_path[-3:] == 'pos':
                    documents.append((words,'pos'))
                elif file_path[-3:] == 'neg':
                    documents.append((words,'neg'))

get_document('aclImdb_v1/aclImdb/train/pos')
get_document('aclImdb_v1/aclImdb/train/neg')

#构建文档-单词矩阵
features = np.zeros([len(documents), len(feature_words)], dtype = float)
for i in range(len(documents)):
    document_words = set(documents[i][0])
    for j in range(len(feature_words)):
        features[i, j] = 1 if (feature_words[j] in document_words) else 0

# 单个将文本表示为布尔特征
def get_document_feature(document):
    document_words = set(document)
    features = np.zeros([1, len(feature_words)], dtype = float)
    for j in range(len(feature_words)):
        features[0, j] = 1 if (feature_words[j] in document_words) else 0
    return features

#print(len(documents))

target = [c for (d, c) in documents]
train_X = features[:18000, :]
train_Y = target[:18000]
test_X = features[18000:, :]
test_Y = target[18000:]

from sklearn import svm
classifier = svm.SVC(kernel = 'rbf')  # kernel为核函数类型，默认RBF径向基神经网络
classifier.fit(train_X, train_Y)

print('支持向量机(SVM)的测试集正确率为', classifier.score(test_X, test_Y))

text = input('请输入影评文本: ')
print('情感分析结果为(pos/neg): ', classifier.predict(get_document_feature(text.split(' '))))
# 注意get_document_feature的参数需要是个词列表，因此需要提前分词，这里使用了最偷工减料的text.split(' ')

