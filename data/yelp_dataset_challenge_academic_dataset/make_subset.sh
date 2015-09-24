#!/bin/bash

subset_size=$1
number_lines=1569264
number_whole_subsets=$(($number_lines/$subset_size))
remainder=$((${number_lines}-(${number_whole_subsets} * ${subset_size})))

review_file="/home/danielj/insight/project/data/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json"
out_file_path="/home/danielj/insight/project/data/yelp_dataset_challenge_academic_dataset/yelp_review_subsets_$subset_size"
mkdir -p $out_file_path

echo $review_file
echo $number_whole_subsets
echo $remainder

for i in `seq 1 $number_whole_subsets`
do
  out_file="${out_file_path}/${i}.json"
  head_num=$((${i} * ${subset_size}))
  echo "head_num: $head_num"
  head -n $head_num $review_file | tail -n $subset_size > $out_file
  echo $out_file
done

out_file="${out_file_path}/$((${number_whole_subsets}+1)).json"
tail -n ${remainder} $review_file > $out_file
echo $out_file
