import argparse

from pytorch_pretrained_bert import BertTokenizer, BertConfig

ENCODE = 'encode'
DECODE = 'decode'

parser = argparse.ArgumentParser(description="Tokenize a text with BERT")
parser.add_argument('-config', choices=[ENCODE, DECODE], help="Specify whether you are converting from a raw sentence to bert tokens or bert tokens to a raw english sentence.")
parser.add_argument('-i', help="The input file")
parser.add_argument('-o', help="The path for the tokenized text file.")

args = parser.parse_args()
input_file = args.i
output_file = args.o
config = args.config


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

with open(input_file) as fp:
    # do stuff
    lines = [line.rstrip('\n') for line in fp]

prev_progress = 0

with open(output_file, 'w+') as fo:
    for index, line in enumerate(lines):
        current_progress = index * 100 // len(lines)
        if current_progress > prev_progress:
            print("{}% done of {} sentences.".format(current_progress, len(lines)))
            prev_progress = current_progress
        
        if config == ENCODE:
            bert_appended_sent = "[CLS] " + line + " [SEP]"
            bert_tokenized = tokenizer.tokenize(bert_appended_sent)
            # join on whitespace
            joined_bert_sent = " ".join(bert_tokenized)
            fo.write("{}\n".format(joined_bert_sent))
            continue
        else:
            # DECODE
            removed_special_tokens = line.replace(' ##', '').replace("[CLS] ", '').replace(" [SEP]", '')
            fo.write("{}\n".format(removed_special_tokens))
            continue
            
