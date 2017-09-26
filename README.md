# deep-ctr
deep models for click through rate prediction

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

# Demo

## Transform text file into TFRecord
```bash
cd demo
sh trans_data_to_tfrecord.sh
cd ..
```

## Train model
```bash
cd demo
sh train.sh
cd ..
```

## Predict using C++
```bash
cd demo
sh test.sh
cd ..
```

# Reference
* [Various Models implemented in TensorFlow](https://github.com/formath/tensorflow-models)
* [Loading a TensorFlow graph with the C++ API](https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f)
* [Loading a tensorflow graph with the C++ API by using Mnist](http://jackytung8085.blogspot.jp/2016/06/loading-tensorflow-graph-with-c-api-by.html)
* [Tensorflow Different ways to Export and Run graph in C++](https://stackoverflow.com/questions/35508866/tensorflow-different-ways-to-export-and-run-graph-in-c/43639305#43639305)

