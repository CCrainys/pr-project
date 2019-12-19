参数包括：num,if_stop_word,word_to_vector,seq_len,word_dim,num_filters,l2_num,learning_rate,dropout_conv,filter_sizes, batch_size,epoch
其中Num现在稳定为10000，无需改，word_to_vector目前是numpy.load(word_to_vector)，seq_len在调整文件时需要根据词向量文件改动

命令举例：
python experiment.py --if_stop_word=0 --seq_len=60 --num_filters=120 --dropout_conv=0.3 --filter_size=2,3,4

输出结果在result.csv内