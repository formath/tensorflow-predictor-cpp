#!/usr/bin/env bash

python2 ../train.py \
    --dict dict.data \
    --continuous_fields "0" \
    --sparse_fields "9,6,116" \
    --linear_fields "152,179" \
    --train_file "libfm.tfrecord" \
    --valid_file "libfm.tfrecord"