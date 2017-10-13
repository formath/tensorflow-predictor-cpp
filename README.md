# deep-ctr
Deep models for click through rate prediction

# Build

## Build TensorFlow
Follow the instruction [build tensorflow from source](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile)
```bash
git clone --recursive https://github.com/tensorflow/tensorflow.git
cd tensorflow
sh tensorflow/contrib/makefile/build_all_ios.sh (depends on your platform)
cd ..
```

## Build deep-ctr
```bash
git clone https://github.com/formath/deep-ctr.git
cd deep-ctr
mkdir build && cd build
cmake ..
make
```

# Simple Demo
This demo used `c=a+b` to show how to save the model and load it using C++ for prediction. [tensorflow_c++_api_prediction_basic](http://mathmach.com/2017/10/09/tensorflow_c++_api_prediction_basic/)
```bash
cd demo/simple_model
# train
sh train.sh
# predict
sh predict.sh
```

# Deep CTR Model Demo
This demo show a real-wrold deep model usage in click through rate prediction. [tensorflow_c++_api_prediction_advance](http://mathmach.com/2017/10/11/tensorflow_c++_api_prediction_advance/)

## Transform LibFM data into TFRecord
* LibFM format: `label fieldId:featureId:value ...`
```bash
cd demo/deep_model
sh trans_data_to_tfrecord.sh
```

## Train model
```bash
sh train.sh
```

## Predict using C++
```bash
sh predict.sh
```

# Reference
* [Various Models implemented in TensorFlow](https://github.com/formath/tensorflow-models)
* [Loading a TensorFlow graph with the C++ API](https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f)
* [Loading a tensorflow graph with the C++ API by using Mnist](http://jackytung8085.blogspot.jp/2016/06/loading-tensorflow-graph-with-c-api-by.html)
* [Tensorflow Different ways to Export and Run graph in C++](https://stackoverflow.com/questions/35508866/tensorflow-different-ways-to-export-and-run-graph-in-c/43639305#43639305)
* [Error issues](https://github.com/tensorflow/tensorflow/issues/3308)

