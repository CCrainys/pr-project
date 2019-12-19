import os

for seq_len in ["30","50"]:
    for num_filters in ["100","200","300"]:
        for filter_size in ["2,3,4","5,6,7"]:
            for dropout in ["0.1","0.3","0.5","0.7"]:
                command="python3 experiment_bert.py --if_stop_word=1 --seq_len=%s --num_filters=%s --dropout_conv=%s --filter_size=%s"%(seq_len,num_filters,dropout,filter_size)
                os.system(command)
