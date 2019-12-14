#!/bin/bash
x=true
while true
do
	read -p "Enter the sentence you want to neutralize subjective bias from: " sent
	echo $sent > tmp.txt
	spm_encode --model=decode.model --output_format=piece < tmp.txt > tmp1.txt
	cat tmp1.txt
	echo "bleh"
	python OpenNMT-py/translate.py -model _step_134520.pt -src tmp1.txt -output pred.txt -verbose
	spm_decode --model=decode.model --input_format=piece < pred.txt > final.txt
        cat final.txt
done	

