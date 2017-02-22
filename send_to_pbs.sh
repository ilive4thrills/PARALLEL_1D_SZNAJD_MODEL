#!/bin/bash

list=$(ls -1 PBS*)

for file in $list
do 
	qsub $file
	sleep 5s
done

# OR

#echo "${list[@]} \n"
