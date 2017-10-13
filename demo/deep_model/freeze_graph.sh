#!/usr/bin/env bash

#MODEL_DIR=`pwd`
#echo $MODEL_DIR
#TensorFlow_HOME=/Users/liujinpeng02/github/tensorflow
#cd ${TensorFlow_HOME}
#python tensorflow/python/tools/freeze_graph.py \
#	--input_graph=${MODEL_DIR}/model/graph.pb \
#    --input_checkpoint=${MODEL_DIR}/model/model.ckeckpoint \
#    --output_graph=${MODEL_DIR}/model/freeze_graph.pb \
#    --output_node_names=cross_entropy
#cd -

python ../../python/freeze_graph.py \
    --model_dir=./saved_model \
    --output_node_names=Softmax