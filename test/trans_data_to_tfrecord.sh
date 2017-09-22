#!/usr/bin/env bash

# generate field dict
python2 ../dict.py '0' '9,6,116' '152,179' libfm.data dict.data

# transform libfm data into tfrecord
python2 ../dict.py dict.data '0' '9,6,116' '152,179' libfm.data libfm.tfrecord