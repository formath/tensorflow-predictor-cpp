#!/usr/bin/env bash

#  Official. Don't know why not works
#MODEL_DIR=`pwd`
#echo $MODEL_DIR
#TensorFlow_HOME=/Users/formath/github/tensorflow
#cd ${TensorFlow_HOME}
#python tensorflow/python/tools/freeze_graph.py \
#	--input_graph=${MODEL_DIR}/model/graph.pb \
#    --input_checkpoint=${MODEL_DIR}/checkpoint/model.ckeckpoint \
#    --output_graph=${MODEL_DIR}/model \
#    --output_node_names='predict/add'
#cd -

# Official. Don't know why not works
#python ../../python/freeze.py \
#    --checkpoint_dir='./checkpoint' \
#    --graph_pb='./model/predict_graph.pb' \
#    --output_node_names='predict/add' \
#    --output_pb='./model/freeze.pb'

# Hack. This works
python ../../python/freeze_graph.py \
    --checkpoint_dir='./checkpoint' \
    --graph_pb='./model/predict_graph.pb' \
    --output_node_names='pctr' \
    --output_dir='./model'
