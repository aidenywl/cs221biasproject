# Preprocessing Step

```shell
python OpenNMT-py/preprocess.py -train_src dataset/neutral/src_train.txt -train_tgt dataset/neutral/tgt_train.txt -valid_src dataset/neutral/src_dev.txt -valid_tgt dataset/neutral/tgt_dev.txt -save_data preprocessed_data/neutral/ --tgt_vocab_size 100000 --src_vocab_size 100000 -report_every 10000
```

```shell
python OpenNMT-py/preprocess.py -train_src dataset/biased_unbiased/src_train.txt -train_tgt dataset/biased_unbiased/tgt_train.txt -valid_src dataset/biased_unbiased/src_dev.txt -valid_tgt dataset/biased_unbiased/tgt_dev.txt -save_data preprocessed_data/biased_unbiased/ --tgt_vocab_size 100000 --src_vocab_size 100000 -report_every 10000
```

# Training Step

```shell
python OpenNMT-py/train.py -data preprocessed_data/neutral/ -save_model models/neutral/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "neutral_log"
```

```shell
python OpenNMT-py/train.py -data preprocessed_data/biased_unbiased/ -save_model models/biased_unbiased/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "biased_unbiasd_log"
```
