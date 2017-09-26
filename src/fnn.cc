#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

using namespace tensorflow;

/*
  // Input data format: libfm data
  // Create a batch of 8 examples having two sparse int and
  // one sparse float features.
  // The data looks like the following:
  // Instance | SparseField | SparseFeatureId | SparseFeatureVal |
  // 0        |    1, 8     |     384, 734    |     1.0, 1.0     |
  // 1        |    3        |     73          |     1.0			 |
  // 2        |             |                 |					 |
  // 3        |    2, 0     |     449, 31     |     1.0, 1.0	 |
  // 4        |             |                 |					 |
  // 5        |             |                 |					 | 
  // 6        |             |                 |					 |
  // 7        |    5        |     465         |     1.0			 |
  // SparseTensor for field id, each SparseTensor construtct of three Tensor
  auto dense_int_indices1 =
      test::AsTensor<int64>({0, 0, 0, 1, 1, 0, 3, 0, 3, 1, 7, 0}, {6, 2}); // 每个数据对应[i,j]，共有6个数字，所以6行，每行最大2个数字，所以2列
  auto dense_int_values1 = test::AsTensor<int64>({1, 8, 0, 2, 0, 5}); // row-major
  auto dense_int_shape1 = TensorShape({8, 2}); // [样本量，每个样本最大id数目]
  sparse::SparseTensor sparse_tensor1(
      dense_int_indices1, dense_int_values1, dense_int_shape1);
   // SparseTensor for feature id
  auto dense_int_indices2 =
      test::AsTensor<int64>({0, 0, 0, 1, 1, 0, 3, 0, 3, 1, 7, 0}, {6, 2});
  auto dense_int_values2 = test::AsTensor<int64>({384, 734, 73, 449, 31, 465});
  auto dense_int_shape2 = TensorShape({8, 2});
  sparse::SparseTensor sparse_tensor2(
      dense_int_indices2, dense_int_values2, dense_int_shape2);
  // SparseTensor for feature value
  auto dense_int_indices3 =
      test::AsTensor<int64>({0, 0, 0, 1, 1, 0, 3, 0, 3, 1, 7, 0}, {6, 2});
  auto dense_float_values3 = test::AsTensor<float32>({1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  auto dense_int_shape3 = TensorShape({8, 2});
  sparse::SparseTensor sparse_tensor3(
      dense_int_indices3, dense_float_values3, dense_int_shape3);
*/
  
int main(int argc, char* argv[]) {
  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Read in the protobuf graph we exported
  // (The path seems to be relative to the cwd. Keep this in mind
  // when using `bazel run` since the cwd isn't where you call
  // `bazel run` but from inside a temp folder.)
  GraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), "../demo/model/graph.pb", &graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Setup inputs and outputs:

  // Our graph doesn't require any inputs, since it specifies default values,
  // but we'll change an input to demonstrate.
  Tensor a(DT_FLOAT, TensorShape());
  a.scalar<float>()() = 3.0;

  Tensor b(DT_FLOAT, TensorShape());
  b.scalar<float>()() = 2.0;

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
    { "a", a },
    { "b", b },
  };

  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  // Run the session, evaluating our "c" operation from the graph
  status = session->Run(inputs, {"c"}, {}, &outputs);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Grab the first output (we only evaluated one graph node: "c")
  // and convert the node to a scalar representation.
  auto output_c = outputs[0].scalar<float>();

  // (There are similar methods for vectors and matrices here:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

  // Print the results
  std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
  std::cout << output_c() << "\n"; // 30

  // Free any resources used by the session
  session->Close();
  return 0;
}