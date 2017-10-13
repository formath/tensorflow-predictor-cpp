#!/usr/bin/env bash

# generate field dict
python ../../python/dict.py \
	'0' \
	'9,6,116' \
	'152,179' \
	./data/libfm.data \
	./data/dict.data

# transform libfm data into tfrecord
python ../../python/data.py \
	./data/dict.data \
	'0' \
	'9,6,116' \
	'152,179' \
	./data/libfm.data \
	./data/libfm.tfrecord
