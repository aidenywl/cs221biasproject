#!/bin/bash
x=true
while true
do
	read -p "Enter the sentence you want to neutralize subjective bias from: " sent
	echo $sent > tmp.txt
	python OpenNMT-py/translate.py -model _step_67260.pt -src tmp.txt -output pred.txt -replace_unk -verbose
        cat pred.txt
done	

