# Preprocessing Step

```shell
python OpenNMT-py/preprocess.py -train_src dataset/neutral/src_train.txt -train_tgt dataset/neutral/tgt_train.txt -valid_src dataset/neutral/src_dev.txt -valid_tgt dataset/neutral/tgt_dev.txt -save_data preprocessed_data/neutral/ --tgt_vocab_size 100000 --src_vocab_size 100000 -report_every 10000
```

```shell
python OpenNMT-py/preprocess.py -train_src dataset/biased_unbiased/biased_src_train.txt -train_tgt dataset/biased_unbiased/biased_tgt_train.txt -valid_src dataset/biased_unbiased/biased_src_dev.txt -valid_tgt dataset/biased_unbiased/biased_tgt_dev.txt -save_data preprocessed_data/biased_unbiased/ --tgt_vocab_size 100000 --src_vocab_size 100000 -report_every 10000
```

# Training Step

```shell
python OpenNMT-py/train.py -data preprocessed_data/neutral/ -save_model ~/models/neutral/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 300000 --log_file "neutral_log" --tensorboard --tensorboard_log_dir="./pretrain_logs/" --batch_size 16 --valid_steps 23500 --save_checkpoint_steps 11750
```

```shell
python OpenNMT-py/train.py -data preprocessed_data/biased_unbiased/ -save_model ~/models/biased_unbiased/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "biased_unbiased_log" --tensorboard --tensorboard_log_dir="./train_logs/" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16
```

# More vocab

```shell
python OpenNMT-py/preprocess.py -train_src dataset/biased_unbiased/biased_src_train.txt -train_tgt dataset/biased_unbiased/biased_tgt_train.txt -valid_src dataset/biased_unbiased/biased_src_dev.txt -valid_tgt dataset/biased_unbiased/biased_tgt_dev.txt -save_data preprocessed_data/biased_100000/ --tgt_vocab_size 100000 --src_vocab_size 100000 -report_every 10000 --share_vocab
```

```shell
python OpenNMT-py/train.py -data preprocessed_data/biased_64000/ -save_model ~/models/biased_64000/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "logs/biased_64000" --tensorboard --tensorboard_log_dir="./train_logs/" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16 --global_attention mlp
```
