#!/usr/bin/env bash

# generate field dict
python ../python/dict.py '0' '9,6,116' '152,179' libfm.data dict.data

# transform libfm data into tfrecord
python ../python/data.py dict.data '0' '9,6,116' '152,179' libfm.data libfm.tfrecord
