#!/usr/bin/env bash

# train
# to save model and checkpoint
python ../../python/train.py \
    --sparse_fields "9,6,116,152,179" \
    --train_file "./data/libfm.tfrecord" \
    --valid_file "./data/libfm.tfrecord"

# just save a model same with train
# except tf.Example input part replaced by placeholder
# for feed Tensor when prediction
python ../../python/predict_model.py \
    --sparse_fields "9,6,116,152,179"
