import common
import keras
from sklearn.model_selection import KFold

## extra imports to set GPU options
from keras import backend as k
import warnings

import argparse
import numpy


parser = argparse.ArgumentParser(description='manual to this script')#传入参数 ，包括数据集数量、是否使用停用词、词向量模型、句子长度、词向量维度、卷积核数量、学习率、drop_out率、不同种类卷积核大小、batch_size,epoch

parser.add_argument('--num', type=int, default = 10000)
parser.add_argument('--if_stop_word', type=int, default = 0)
#sparser.add_argument('--word_to_vector', type=str, default = 'Word2VecModel')
parser.add_argument('--seq_len', type=int, default = 50)
parser.add_argument('--word_dim', type=int, default = 100)
parser.add_argument('--num_filters', type=int, default = 100)
parser.add_argument('--l2_num', type=float, default = 0.001)
parser.add_argument('--learning_rate', type=float, default = 1e-3)
parser.add_argument('--dropout_conv', type=float, default = 0.5)
parser.add_argument('--filter_size', type=str, default = '1,2,3')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=50)
args = parser.parse_args()

num,if_stop_word,seq_len,word_dim,num_filters,l2_num,learning_rate,dropout_conv,filter_sizes, batch_size,epoch = common.load_parameter(args)
print(num,if_stop_word,seq_len,word_dim,num_filters,l2_num,learning_rate,dropout_conv,filter_sizes, batch_size,epoch)

word_to_vector="word2vec_"+str(word_dim)+"_"+str(seq_len)+".npy"

warnings.filterwarnings("ignore")

#读入数据
data = common.load_data('train.csv')
if if_stop_word:
    stop_word = common.load_stop_word('stopword.txt')
else:
    stop_word = []

#得到标签 词向量,不需要改，这部分得到label，vector之后会被覆盖
label, vector = common.word_to_vector('Word2VecModel',data[:num], seq_len, 100, stop_word)

#得到词向量，加载模型为vector
vector = numpy.load(word_to_vector)
print(label.shape, vector.shape)

MODELS={
	'cnn':lambda:common.get_cnn(
							input_shape=[seq_len, word_dim],
							num_outputs=1,
                            l2_num = l2_num,
							num_filters=num_filters,
                            filter_sizes = filter_sizes,
							learning_rate=learning_rate,
							dropout_conv=dropout_conv,)
}


early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience = 5,
                              verbose=0, mode='auto')   #20个回合内 min_delta为0 ，则 停止


num_splits = 10   #k交叉验证
kf_iterator = KFold(n_splits=num_splits, shuffle=True, random_state=42)


pre = []
recall = []
f1 = []
for i, splits in enumerate(kf_iterator.split(vector)):
    train, test = splits
    k.clear_session()
    for model_name, model_info in MODELS.items():
        print(model_name)
        model=model_info()

        data_train = vector[train]
        label_train = label[train]
        data_test = vector[test]
        label_test = label[test]
        print(data_train.shape,label_train.shape)

        if model_name=='cnn':
            model.fit(	data_train,
						label_train,
						epochs=epoch,
						batch_size=batch_size,
						validation_split=.1,
						callbacks=[early_stopping])

        else:
            raise ValueError('there is something wrong')

    pred = model.predict(data_test)
    pred = common.modify_pred(pred)
    cur_pre,cur_recall,cur_f1 = common.output_result(pred.flatten(), label_test.flatten())
    print('当前为第',i,'部分，该部分数据为：', cur_pre,cur_recall,cur_f1)
    pre.append(cur_pre)
    recall.append(cur_recall)
    f1.append(cur_f1)

print('pre\n',pre,'\nrecall\n',recall,'\nf1\n',f1)
final_pre = sum(pre)/num_splits
final_recall = sum(recall)/num_splits
final_f1 = sum(f1)/num_splits
print('最终数据为(pre recall f1)',final_pre,final_recall,final_f1)

str1 = str(num) + ' '+str(if_stop_word)+' '+str(word_to_vector)+' '+str(seq_len)+' '+str(word_dim)+' '+str(num_filters)+' '+str(l2_num)+' '+str(learning_rate)+' '+str(dropout_conv)+' '+str(filter_sizes)+' '+str(batch_size)+' '+str(epoch)
str2 = str(final_pre)+' '+str(final_recall)+' '+str(final_f1)
print(str1,str2)
common.load_result(str1,str2)



