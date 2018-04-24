#!/usr/bin/env bash

# transform libfm data into tfrecord
python ../../python/data.py \
	'9,6,116,152,179' \
	./data/libfm.data \
	./data/libfm.tfrecord

if [[ $? != 0 ]]; then
	echo "generate tfrecord error" && exit 1
fi