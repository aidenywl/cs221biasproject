# Analyzing Tokenizers for neutralizing bias in text.

Stanford University CS221 Introduction to Artificial Intelligence p-final.

Out of all the models we have trained, the OpenNMT model trained with:
- OpenNMT's standard white space and punctuation tokenizer
- A vocabulary size of 32000
- A seed of -1
- At 25 epochs

Achieved the best accuracy of 32.70% on our test set.

The OpenNMT mode can be downloaded here: https://drive.google.com/file/d/1Stl_TqsTHfua2Fey0lx18vmJfdpbpzIN/view?usp=sharing

# Setting up the server
The OpenNMT flask server is set up and extended with CORS to allow it to be used like an API by any host.

The server and model configuration is in `OpenNMT-py/available_models/conf.json`

To start the server, simply run
```
export IP="0.0.0.0"
export PORT=5000
export URL_ROOT="/translator"
export CONFIG="./available_models/conf.json"

python server.py --ip $IP --port $PORT --url_root $URL_ROOT -- config $CONFIG
```


# Preprocessing Step

```shell
python OpenNMT-py/preprocess.py -train_src dataset/neutral/src_train.txt -train_tgt dataset/neutral/tgt_train.txt -valid_src dataset/neutral/src_dev.txt -valid_tgt dataset/neutral/tgt
_dev.txt -save_data preprocessed_data/neutral/ --tgt_vocab_size 32000 --src_vocab_size 32000 -report_every 10000
```

# 32000

```shell
python OpenNMT-py/preprocess.py -train_src dataset/biased_nmt/train.biased -train_tgt dataset/biased_nmt/train.unbiased -valid_src dataset/biased_nmt/dev.biased -valid_tgt dataset/biased_nmt/dev.unbiased -save_data preprocessed_data/standard_32000/ --tgt_vocab_size 32000 --src_vocab_size 32000 -report_every 10000 --share_vocab
```

```shell
python OpenNMT-py/train.py -data preprocessed_data/standard_32000/ -save_model ~/models/standard_32000/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "logs/standard_32000" --tensorboard --tensorboard_log_dir="./train_logs/standard_32000" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16 --global_attention mlp --src_word_vec_size 512 --tgt_word_vec_size 512 --enc_rnn_size 512 --dec_rnn_size 512 --seed -1
```

# 64000

```shell
python OpenNMT-py/preprocess.py -train_src dataset/biased_nmt/train.biased -train_tgt dataset/biased_nmt/train.unbiased -valid_src dataset/biased_nmt/dev.biased -valid_tgt dataset/biased_nmt/dev.unbiased -save_data preprocessed_data/standard_64000/ --tgt_vocab_size 64000 --src_vocab_size 64000 -report_every 10000 --share_vocab
```

python OpenNMT-py/train.py -data preprocessed_data/standard_64000/ -save_model ~/models/standard_64000/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "logs/standard_64000" --tensorboard --tensorboard_log_dir="./train_logs/" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16 --global_attention mlp --src_word_vec_size 512 --tgt_word_vec_size 512 --enc_rnn_size 512 --dec_rnn_size 512 --seed -1

# BRNN

python OpenNMT-py/train.py -data preprocessed_data/standard_64000/ -save_model ~/models/standard_64000_brnn/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "logs/standard_64000_brnn" --tensorboard --tensorboard_log_dir="./train_logs/standard_64000_brnn" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16 --global_attention mlp --src_word_vec_size 512 --tgt_word_vec_size 512 --enc_rnn_size 512 --dec_rnn_size 512 --seed -1

# 32000 BPE

spm_train --input=biased_nmt/all.txt --model_prefix=bpe32000 --vocab_size=32000 --character_coverage=1.0 --model_type=bpe

spm_encode --model=final_bpe_32000/bpe32000.model --output_format=piece < biased_nmt/train.biased > final_bpe_32000/train.biased

spm_encode --model=final_bpe_32000/bpe32000.model --output_format=piece < biased_nmt/train.unbiased > final_bpe_32000/train.unbiased

spm_encode --model=final_bpe_32000/bpe32000.model --output_format=piece < biased_nmt/dev.biased > final_bpe_32000/dev.biased

spm_encode --model=final_bpe_32000/bpe32000.model --output_format=piece < biased_nmt/dev.unbiased > final_bpe_32000/dev.unbiased

spm_encode --model=dataset/final_bpe_32000/bpe32000.model --output_format=piece < dataset/biased_nmt/test.biased > dataset/final_bpe_32000/test.biased

spm_encode --model=dataset/final_bpe_32000/bpe32000.model --output_format=piece < dataset/biased_nmt/test.unbiased > dataset/final_bpe_32000/test.unbiased

python OpenNMT-py/preprocess.py -train_src dataset/final_bpe_32000/train.biased -train_tgt dataset/final_bpe_32000/train.unbiased -valid_src dataset/final_bpe_32000/dev.biased -valid_tgt dataset/final_bpe_32000/dev.unbiased -save_data preprocessed_data/final_bpe_32000/ --src_vocab dataset/final_bpe_32000/bpe32000.vocab --tgt_vocab dataset/final_bpe_32000/bpe32000.vocab --share_vocab

python OpenNMT-py/train.py -data preprocessed_data/final_bpe_32000/ -save_model ~/models/final_bpe_32000/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "logs/final_bpe_32000" --tensorboard --tensorboard_log_dir="./train_logs/" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16 --global_attention mlp --src_word_vec_size 512 --tgt_word_vec_size 512 --enc_rnn_size 512 --dec_rnn_size 512 --seed -1

# 64000 BPE

spm_encode --model=final_bpe_64000/bpe64000.model --output_format=piece < biased_nmt/train.biased > final_bpe_64000/train.biased

spm_encode --model=final_bpe_64000/bpe64000.model --output_format=piece < biased_nmt/train.unbiased > final_bpe_64000/train.unbiased

spm_encode --model=final_bpe_64000/bpe64000.model --output_format=piece < biased_nmt/dev.biased > final_bpe_64000/dev.biased

spm_encode --model=final_bpe_64000/bpe64000.model --output_format=piece < biased_nmt/dev.unbiased > final_bpe_64000/dev.unbiased

spm_encode --model=dataset/final_bpe_64000/bpe64000.model --output_format=piece < dataset/biased_nmt/test.biased > dataset/final_bpe_64000/test.biased

spm_encode --model=dataset/final_bpe_64000/bpe64000.model --output_format=piece < dataset/biased_nmt/test.unbiased > dataset/final_bpe_64000/test.unbiased

python OpenNMT-py/preprocess.py -train_src dataset/final_bpe_64000/train.biased -train_tgt dataset/final_bpe_64000/train.unbiased -valid_src dataset/final_bpe_64000/dev.biased -valid_tgt dataset/final_bpe_64000/dev.unbiased -save_data preprocessed_data/final_bpe_64000/ --src_vocab dataset/final_bpe_64000/bpe64000.vocab --tgt_vocab dataset/final_bpe_64000/bpe64000.vocab --share_vocab

python OpenNMT-py/train.py -data preprocessed_data/final_bpe_64000/ -save_model ~/models/final_bpe_64000/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "logs/final_bpe_64000" --tensorboard --tensorboard_log_dir="./train_logs/" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16 --global_attention mlp --src_word_vec_size 512 --tgt_word_vec_size 512 --enc_rnn_size 512 --dec_rnn_size 512 --seed -1

# Training Step

```shell
python OpenNMT-py/train.py -data preprocessed_data/neutral/ -save_model ~/models/neutral/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 300000 --log_file "neutral_log" --tensorboard --tensorboard_log_dir="./pretrain_logs/" --batch_size 16 --valid_steps 23500 --save_checkpoint_steps 11750
```

```shell
python OpenNMT-py/train.py -data preprocessed_data/biased_unbiased/ -save_model ~/models/biased_unbiased/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "biased_unbiased_log" --tensorboard --tensorboard_log_dir="./train_logs/" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16
```

# Seed 42

python OpenNMT-py/train.py -data preprocessed_data/standard_64000/ -save_model ~/models/standard_64000_seed42/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "logs/standard_64000_seed42" --tensorboard --tensorboard_log_dir="./train_logs/standard_64000_seed42" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16 --global_attention mlp --src_word_vec_size 512 --tgt_word_vec_size 512 --enc_rnn_size 512 --dec_rnn_size 512 --seed 42

# ################################### BIDIRECTIONAL

# 32000

```shell
python OpenNMT-py/train.py -data preprocessed_data/standard_32000/ -save_model ~/models/standard_32000_brnn/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "logs/standard_32000_brnn" --tensorboard --tensorboard_log_dir="./train_logs/standard_32000_brnn" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16 --global_attention mlp --src_word_vec_size 512 --tgt_word_vec_size 512 --enc_rnn_size 512 --dec_rnn_size 512 --seed -1 --encoder_type brnn
```

# 64000

```shell
python OpenNMT-py/train.py -data preprocessed_data/standard_64000/ -save_model ~/models/standard_64000_brnn/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "logs/standard_64000_brnn" --tensorboard --tensorboard_log_dir="./train_logs/standard_64000_brnn" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16 --global_attention mlp --src_word_vec_size 512 --tgt_word_vec_size 512 --enc_rnn_size 512 --dec_rnn_size 512 --seed -1 --encoder_type brnn
```

# 32000 BPE

python OpenNMT-py/train.py -data preprocessed_data/final_bpe_32000/ -save_model ~/models/final_bpe_32000_brnn/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "logs/final_bpe_32000_brnn" --tensorboard --tensorboard_log_dir="./train_logs/final_bpe_32000_brnn" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16 --global_attention mlp --src_word_vec_size 512 --tgt_word_vec_size 512 --enc_rnn_size 512 --dec_rnn_size 512 --seed -1 --encoder_type brnn

# 64000 BPE

python OpenNMT-py/train.py -data preprocessed_data/final_bpe_64000/ -save_model ~/models/final_bpe_64000_brnn/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "logs/final_bpe_64000_brnn" --tensorboard --tensorboard_log_dir="./train_logs/final_bpe_64000_brnn" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16 --global_attention mlp --src_word_vec_size 512 --tgt_word_vec_size 512 --enc_rnn_size 512 --dec_rnn_size 512 --seed -1 --encoder_type brnn

# SEED FORTY TWO

# 32000

```shell
python OpenNMT-py/train.py -data preprocessed_data/standard_32000/ -save_model ~/models/standard_32000_brnn_42/  -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "logs/standard_32000_brnn_42" --tensorboard --tensorboard_log_dir="./train_logs/standard_32000_brnn_42" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16 --global_attention mlp --src_word_vec_size 512 --tgt_word_vec_size 512 --enc_rnn_size 512 --dec_rnn_size 512 --seed 42 --encoder_type brnn
```

# 64000

```shell
python OpenNMT-py/train.py -data preprocessed_data/standard_64000/ -save_model ~/models/standard_64000_brnn_42/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "logs/standard_64000_brnn_42" --tensorboard --tensorboard_log_dir="./train_logs/standard_64000_brnn_42" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16 --global_attention mlp --src_word_vec_size 512 --tgt_word_vec_size 512 --enc_rnn_size 512 --dec_rnn_size 512 --seed 42 --encoder_type brnn
```

```shell
# 32000 BPE

python OpenNMT-py/train.py -data preprocessed_data/final_bpe_32000/ -save_model ~/models/final_bpe_32000_brnn_42/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "logs/final_bpe_32000_brnn_42" --tensorboard --tensorboard_log_dir="./train_logs/final_bpe_32000_brnn_42" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16 --global_attention mlp --src_word_vec_size 512 --tgt_word_vec_size 512 --enc_rnn_size 512 --dec_rnn_size 512 --seed 42 --encoder_type brnn

# 64000 BPE

python OpenNMT-py/train.py -data preprocessed_data/final_bpe_64000/ -save_model ~/models/final_bpe_64000_brnn_42/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "logs/final_bpe_64000_brnn_42" --tensorboard --tensorboard_log_dir="./train_logs/final_bpe_64000_brnn_42" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16 --global_attention mlp --src_word_vec_size 512 --tgt_word_vec_size 512 --enc_rnn_size 512 --dec_rnn_size 512 --seed 42 --encoder_type brnn

# BERTTTTTT

python OpenNMT-py/preprocess.py -train_src dataset/bert/bert_train.biased -train_tgt dataset/bert/bert_train.unbiased -valid_src dataset/bert/bert_dev.biased -valid_tgt dataset/bert/bert_dev.unbiased -save_data preprocessed_data/bert/ --src_vocab dataset/bert/vocab.txt --tgt_vocab dataset/bert/vocab.txt --share_vocab

python OpenNMT-py/train.py -data preprocessed_data/bert/ -save_model ~/models/bert_brnn_-1/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "logs/final_final_bert_brnn-1" --tensorboard --tensorboard_log_dir="./train_logs/final_final_bert_brnn-1" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16 --global_attention mlp --src_word_vec_size 512 --tgt_word_vec_size 512 --enc_rnn_size 512 --dec_rnn_size 512 --seed -1 --encoder_type brnn

python OpenNMT-py/train.py -data preprocessed_data/bert/ -save_model ~/models/bert_brnn_42/ -gpu_ranks 0 -learning_rate 0.005 -opt adam -train_steps 500000 --log_file "logs/bert_brnn_42" --tensorboard --tensorboard_log_dir="./train_logs/bert_brnn_42" --save_checkpoint_steps 3363 --valid_steps 3363 --batch_size 16 --global_attention mlp --src_word_vec_size 512 --tgt_word_vec_size 512 --enc_rnn_size 512 --dec_rnn_size 512 --seed 42 --encoder_type brnn


# Test step


# Standard 32000 Seed -1

python OpenNMT-py/translate.py -model ~/models/standard_32000_brnn/_step_84075.pt -src dataset/biased_nmt/test.biased -tgt dataset/biased_nmt/test.unbiased -output test_results/standard_32000_brnn_-1 -gpu 0 -replace_unk

python manual_diff_check.py dataset/biased_nmt/test.unbiased test_results/standard_32000_brnn_-1

perl multi-bleu.perl < dataset/biased_nmt/test.unbiased test_results/standard_32000_brnn_-1

# Standard 64000 Seed -1

python OpenNMT-py/translate.py -model ~/models/standard_64000_brnn/_step_84075.pt -src dataset/biased_nmt/test.biased -tgt dataset/biased_nmt/test.unbiased -output test_results/standard_64000_brnn_-1 -gpu 0 -replace_unk

python manual_diff_check.py dataset/biased_nmt/test.unbiased test_results/standard_64000_brnn_-1

perl multi-bleu.perl < dataset/biased_nmt/test.unbiased test_results/standard_64000_brnn_-1

# BPE 32000 Seed -1

python OpenNMT-py/translate.py -model ~/models/final_bpe_32000_brnn/_step_84075.pt -src dataset/final_bpe_32000/test.biased -tgt dataset/final_bpe_32000/test.unbiased -output test_results/bpe_32000_brnn_-1 -gpu 0 -replace_unk

python manual_diff_check.py dataset/final_bpe_32000/test.unbiased test_results/bpe_32000_brnn_-1

spm_decode --model=dataset/final_bpe_32000/bpe32000.model --input_format=piece < test_results/bpe_32000_brnn_-1 > test_results/bpe_32000_brnn_-1_decoded


perl multi-bleu.perl < dataset/biased_nmt/test.unbiased test_results/bpe_32000_brnn_-1_decoded



# BPE 64000 Seed -1

python OpenNMT-py/translate.py -model ~/models/final_bpe_64000_brnn/_step_84075.pt -src dataset/final_bpe_64000/test.biased -tgt dataset/final_bpe_64000/test.unbiased -output test_results/bpe_64000_brnn_-1 -gpu 0 -replace_unk

python manual_diff_check.py dataset/final_bpe_64000/test.unbiased test_results/bpe_64000_brnn_-1

spm_decode --model=dataset/final_bpe_32000/bpe32000.model --input_format=piece < test_results/bpe_32000_brnn_-1 > test_results/bpe_32000_brnn_-1_decoded


perl multi-bleu.perl < dataset/biased_nmt/test.unbiased test_results/bpe_64000_-1_decoded


# BERT Seed 42

python OpenNMT-py/translate.py -model ~/models/bert_brnn_42/_step_84075.pt -src dataset/bert/bert_test.biased -tgt dataset/bert/bert_test.unbiased -output test_results/bert_brnn_42 -gpu 0

python manual_diff_check.py dataset/bert/bert_test.unbiased test_results/bert_brnn_42

perl multi-bleu.perl < dataset/bert/bert_test.unbiased test_results/bert_brnn_42



# Standard 32000 Seed 42

python OpenNMT-py/translate.py -model ~/models/standard_32000_brnn_42/_step_84075.pt -src dataset/biased_nmt/test.biased -tgt dataset/biased_nmt/test.unbiased -output test_results/standard_32000_brnn_42 -gpu 0 -replace_unk

python manual_diff_check.py dataset/biased_nmt/test.unbiased test_results/standard_32000_brnn_42

perl multi-bleu.perl < dataset/biased_nmt/test.unbiased test_results/standard_32000_brnn_42


# Standard 64000 Seed 42

python OpenNMT-py/translate.py -model ~/models/standard_64000_brnn_42/_step_84075.pt -src dataset/biased_nmt/test.biased -tgt dataset/biased_nmt/test.unbiased -output test_results/standard_64000_brnn_42 -gpu 0 -replace_unk

python manual_diff_check.py dataset/biased_nmt/test.unbiased test_results/standard_64000_brnn_42

perl multi-bleu.perl < dataset/biased_nmt/test.unbiased test_results/standard_64000_brnn_42

# BPE 32000 Seed 42

python OpenNMT-py/translate.py -model ~/models/final_bpe_32000_brnn_42/_step_84075.pt -src dataset/final_bpe_32000/test.biased -tgt dataset/final_bpe_32000/test.unbiased -output test_results/bpe_32000_brnn_42 -gpu 0 -replace_unk

python manual_diff_check.py dataset/final_bpe_32000/test.unbiased test_results/bpe_32000_brnn_42

spm_decode --model=dataset/final_bpe_32000/bpe32000.model --input_format=piece < test_results/bpe_32000_brnn_42 > test_results/bpe_32000_brnn_42_decoded


perl multi-bleu.perl < dataset/biased_nmt/test.unbiased test_results/bpe_32000_brnn_42_decoded

perl multi-bleu.perl < dataset/final_bpe_32000/test.unbiased test_results/bpe_32000_brnn_42



# BPE 64000 Seed 42

python OpenNMT-py/translate.py -model ~/models/final_bpe_64000_brnn_42/_step_84075.pt -src dataset/final_bpe_64000/test.biased -tgt dataset/final_bpe_64000/test.unbiased -output test_results/bpe_64000_brnn_42 -gpu 0 -replace_unk

python manual_diff_check.py dataset/final_bpe_64000/test.unbiased test_results/bpe_64000_brnn_42

spm_decode --model=dataset/final_bpe_64000/bpe64000.model --input_format=piece < test_results/bpe_64000_brnn_42 > test_results/bpe_64000_brnn_42_decoded


perl multi-bleu.perl < dataset/biased_nmt/test.unbiased test_results/bpe_64000_brnn_42_decoded
```
