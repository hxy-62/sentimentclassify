import os
import string

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet
from nltk.corpus import wordnet
from utils import clean_str

# 停用词
stpw = stopwords.words('english')
# 标点符号
punc = list(string.punctuation)
# 不需要分析的词和标点
stop = punc + stpw

#将pos_tag得到的词性转化为senti_synsets中要用到的词性
tag_map = {'NN': 'n', 'NNP': 'n', 'NNPS': 'n', 'NNS': 'n', 'UH': 'n',\
           'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v',\
           'JJ': 'a', 'JJR': 'a', 'JJS': 'a',\
           'RB': 'r', 'RBR': 'r', 'RBS': 'r', 'RP': 'r', 'WRB': 'r'}


path1 = 'aclImdb_v1/aclImdb/train/pos'
path2 = 'aclImdb_v1/aclImdb/train/neg'

def cal_acc(folder_path):
    correct = 0
    total = len(os.listdir(folder_path))  
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r',encoding='utf-8') as file:
                sentence = clean_str(file.readline())
                words = word_tokenize(sentence)
                for word in words:
                    if word.lower() in stop:
                        words.remove(word)
                word_tag = pos_tag(words)
                word_tag = [(t[0], tag_map[t[1]]) if t[1] in tag_map else (t[0], '') for t in word_tag]
                sentiment_synsets = [list(sentiwordnet.senti_synsets(t[0], t[1])) for t in word_tag]
                score = sum(sum([x.pos_score() - x.neg_score() for x in s]) / len(s) for s in sentiment_synsets if len(s) != 0)
                if folder_path[-3:] == 'pos':
                    if score > 0 :
                        correct += 1
                elif folder_path[-3:] == 'neg':
                    if score < 0 :
                        correct += 1
    acc = correct / total
    return acc, correct, total

#计算积极样本中的准确率：
pos_acc, pos_correct, pos_total = cal_acc(path1)
#计算消极样本中的准确率：
neg_acc, neg_correct, neg_total = cal_acc(path2)
total_acc = (pos_correct+neg_correct) / (pos_total+neg_total)
print("积极样本中的准确率为: {:.2%},消极样本中的准确率为: {:.2%}, 总的准确率为: {:.2%}".format(pos_acc,neg_acc,total_acc))