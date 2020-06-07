#!/bin/bash
x=true
while true
do
	read -p "Enter the sentence you want to neutralize subjective bias from: " sent
	echo $sent > tmp.txt
	python OpenNMT-py/translate.py -model standard_32000_brnn_seed-1_step_84075_epoch_25.pt -src tmp.txt -output pred.txt -replace_unk -verbose
        cat pred.txt
done	

