from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, Concatenate, Bidirectional, Conv1D, MaxPool1D, Conv2D, \
    MaxPool2D, Reshape, Flatten, AveragePooling1D
from keras.optimizers import Adam
from keras import regularizers
from gensim.models import word2vec
import csv
import pandas as pd
import numpy as np
import jieba
import re
import itertools
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score

def load_result(str1,str2):

    f = open('result.csv', 'a', encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow([str1, str2])
    f.close()

def load_parameter(args): #载入参数
    num = args.num
    stop_word = args.if_stop_word
    word_to_vector = args.word_to_vector
    seq_len = args.seq_len
    word_dim = args.word_dim
    num_filters = args.num_filters
    l2_num = args.l2_num
    learning_rate = args.learning_rate
    dropout_conv = args.dropout_conv
    filter_size = args.filter_size.split(',')
    batch_size = args.batch_size
    epoch = args.epoch
    return num,stop_word,word_to_vector,seq_len,word_dim,num_filters,l2_num,learning_rate,dropout_conv,filter_size, batch_size,epoch


def load_data(path): #载入数据
    data = pd.read_csv(path, sep='\t')
    data = np.array(data)
    return data

def load_stop_word(path):  #载入停用词
    result = []
    with open(path, 'r',  encoding='UTF-8') as fp:
        lines = fp.readlines()
        for line in lines:
            result.append(line.strip())
    return result


def word_to_vector(word_to_vector_path,data, seq_len, word_dim, stop_word):  #输入数据转化为词向量
    label = data[:, :1]
    seq = data[:, -1:]
    vector = []
    model = word2vec.Word2Vec.load(word_to_vector_path)
    print('loading_data......')
    for i in range(len(seq)):
        if i % 500 == 0:
            print(i)
        cur_seq = seq[i]
        cur_vector = []
        cur_seq[0] = re.findall(r'[\u4e00-\u9fa5]', cur_seq[0])
        cur_seq[0] = "".join(itertools.chain(*cur_seq[0]))
        seg_list = jieba.cut(cur_seq[0], cut_all=False)
        words = []
        for word in seg_list:
            if word not in stop_word:
                words.append(word)
        extra = [0 for i in range(word_dim)]
        if len(words) > seq_len:
            for i in range(seq_len):
                try:
                    cur_vector.append(model[words[len(words)-1-seq_len+i]])
                except:
                    cur_vector.append(extra)
        else:
            for i in range(len(words)):
                try:
                    cur_vector.append(model[words[i]])
                except:
                    cur_vector.append(extra)
            for i in range(seq_len-len(words)):
                cur_vector.append(extra)
        cur_vector = np.array(cur_vector)
        vector.append(cur_vector)
    vector = np.array(vector)
    print(vector.shape)
    return label, vector


def get_cnn(input_shape, num_outputs,l2_num, num_filters,filter_sizes, learning_rate, dropout_conv):  # cnn模型构建

    # filter_sizes=[3,4,5]
    embedding_dim = input_shape[1]
    sequence_length = input_shape[0]

    l2_strength = l2_num

    inputs = Input(shape=input_shape)
    inputs_drop = Dropout(dropout_conv)(inputs)

    filter_size = int(filter_sizes[0])
    conv_1 = Conv1D(filters=num_filters, kernel_size=filter_size, strides=1, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_strength))(inputs_drop)  # 卷积size为1 滑动strides为1
    pool_1 = AveragePooling1D(pool_size=input_shape[0] - filter_size + 1, strides=1)(conv_1)  # 均值池化
    pool_drop_1 = Dropout(dropout_conv)(pool_1)

    filter_size = int(filter_sizes[1])
    conv_2 = Conv1D(filters=num_filters, kernel_size=filter_size, strides=1, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_strength))(inputs_drop)
    pool_2 = AveragePooling1D(pool_size=input_shape[0] - filter_size + 1, strides=1)(conv_2)
    pool_drop_2 = Dropout(dropout_conv)(pool_2)

    filter_size = int(filter_sizes[2])
    conv_3 = Conv1D(filters=num_filters, kernel_size=filter_size, strides=1, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_strength))(inputs_drop)
    pool_3 = AveragePooling1D(pool_size=input_shape[0] - filter_size + 1, strides=1)(conv_3)
    pool_drop_3 = Dropout(dropout_conv)(pool_3)

    concatenated = Concatenate(axis=1)([pool_drop_1, pool_drop_2, pool_drop_3])

    dense = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_strength))(
        Flatten()(concatenated))  # 全连接
    dense_drop = Dropout(.5)(dense)

    output = Dense(units=num_outputs, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_strength))(
            dense_drop)

    #create
    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model

def output_result(pred, test_label):  #输出结果
    TN = 0
    FN = 0
    TP = 0
    FP = 0
    for i in range(len(pred)):
        if pred[i] == 1 and test_label[i] == 1:
            TP += 1
        elif pred[i] == 1 and test_label[i] == 0:
            FP += 1
        elif pred[i] == 0 and test_label[i] == 1:
            FN += 1
        else:
            TN += 1
    pre = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = (2*pre*recall)/(pre+recall)
    print("准确率 召回率 F1", pre, recall, f1)
    return pre, recall, f1

def modify_pred(pred):
    result = []
    for data in pred:
        if data > 0.5:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)



