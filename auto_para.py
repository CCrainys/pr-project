import os

for seq_len in ["30","50"]:
    for word_dim in ["100","200","400"]:
        for num_filters in ["100","200","300"]:
            for filter_size in ["2,3,4","5,6,7"]:
                for dropout in ["0.1","0.3","0.5","0.7"]:
                    command="python experiment.py --if_stop_word=1 --seq_len=%s --word_dim=%s  --num_filters=%s --dropout_conv=%s --filter_size=%s"%(seq_len,word_dim,num_filters,dropout,filter_size)
                    print(command)
