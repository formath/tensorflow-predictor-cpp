#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include "util.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

using namespace tensorflow;

/*
  // Input data format: libfm data
  // Create a batch of 8 examples having two sparse int and
  // one sparse float features.
  // The data looks like the following:
  // Instance | SparseField | SparseFeatureId | SparseFeatureVal |
  // 0        |    1, 8     |     384, 734    |     1.0, 1.0     |
  // 1        |    3        |     73          |     1.0          |
  // 2        |             |                 |                  |
  // 3        |    2, 0     |     449, 31     |     1.0, 1.0     |
  // 4        |             |                 |                  |
  // 5        |             |                 |                  |
  // 6        |             |                 |                  |
  // 7        |    5        |     465         |     1.0          |
  // SparseTensor for field id, each SparseTensor construtct of three Tensor
  auto dense_int_indices1 =
      test::AsTensor<int64>({0, 0, 0, 1, 1, 0, 3, 0, 3, 1, 7, 0}, {6, 2});
  auto dense_int_values1 = test::AsTensor<int64>({1, 8, 0, 2, 0, 5}); // row-major
  auto dense_int_shape1 = TensorShape({8, 2});
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
  // parse field
  std::vector<std::string> tokens;
  std::vector<int> sparse_field; // sparse field
  util::split(argv[1], ',', tokens);
  for (std::string token: tokens) {
    sparse_field.push_back(std::stoi(token));
  }
  std::vector<int> linear_field; // linear field
  util::split(argv[2], ',', tokens);
  for (std::string token: tokens) {
    linear_field.push_back(std::stoi(token));
  }
  std::vector<int> continuous_field; // continuous field
  util::split(argv[3], ',', tokens);
  for (std::string token: tokens) {
    continuous_field.push_back(std::stoi(token));
  }

  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  } else {
    std::cout << "Session created successfully" << std::endl;
  }

  // Read in the protobuf graph we exported
  // (The path seems to be relative to the cwd. Keep this in mind
  // when using `bazel run` since the cwd isn't where you call
  // `bazel run` but from inside a temp folder.)
  GraphDef graph_def;
  std::string graph_path = argv[4];
  status = ReadBinaryProto(Env::Default(), graph_path, &graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
  } else {
    std::cout << "Load graph protobuf successfully" << std::endl;
  }

  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    return 1;
  } else {
    std::cout << "Add graph to session successfully" << std::endl;
  }

  // Read parameters from the saved checkpoint
  /*Tensor checkpointPathTensor(DT_STRING, TensorShape());
  std::string checkpoint_path = argv[5];
  checkpointPathTensor.scalar<std::string>()() = checkpoint_path;
  status = session->Run(
          {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
          {},
          {graph_def.saver_def().restore_op_name()},
          nullptr);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    return 1;
  } else {
    std::cout << "Load checkpoint successfully" << std::endl;
  }*/

  // Setup inputs and outputs
  // input 9:283:1 6:384:1 152:384:1
  std::string libfm_data = "9:283:1 6:384:1 152:384:1";
  std::unordered_map<int32, std::unordered_map<int32, float> > instance;
  std::vector<std::string> features;
  util::split(libfm_data, ' ', features);
  for (std::string feature: features) {
    std::vector<std::string> tokens;
    util::split(feature, ':', tokens);
    int32 fieldid;
    int32 featureid;
    float value;
    int i = 0;
    for (std::string token: tokens) {
      if (i == 0) {
        fieldid = std::stoi(token);
      } else if (i == 1) {
        featureid = std::stoi(token);
      } else if (i == 2) {
        value = std::stof(token);
      }
      i++;
    }
    if (instance.find(fieldid) == instance.end()) {
      std::unordered_map<int32, float> f;
      f[featureid] = value;
      instance[fieldid] = f;
    } else {
      instance[fieldid][featureid] = value;
    }
  }

  std::vector<std::pair<std::string, sparse::SparseTensor> > inputs;
  for (int i = 0; i < sparse_field.size(); i++) {
    uint32 fieldid = sparse_field[i];
    std::vector<int32> indice;
    std::vector<int32> fid_list;
    std::vector<float> fval_list;
    if (instance.find(fieldid) != instance.end()) {
      int num = 0;
      for (std::unordered_map<int32, float>::const_iterator iter = instance[fieldid].begin();
          iter != instance[fieldid].end(); iter++) {
        indice.push_back(0);
        indice.push_back(num++);
        fid_list.push_back(iter->first);
        fval_list.push_back(iter->second);
      }
    } else {
      fid_list.push_back(0); // missid
      fval_list.push_back(0.0);
    }
    auto id_indice_tensor =
      test::AsTensor<int32>(indice, {static_cast<int32>(indice.size()/2), 2});
    auto id_list_tensor = test::AsTensor<int32>(fid_list);
    auto id_tensor_shape = TensorShape({1, static_cast<int32>(fid_list.size())});
    sparse::SparseTensor id_sparse_tensor(id_indice_tensor, id_list_tensor, id_tensor_shape);
    auto val_indice_tensor =
      test::AsTensor<int32>(indice, {static_cast<int32>(indice.size()/2), 2});
    auto val_list_tensor = test::AsTensor<float>(fval_list);
    auto val_tensor_shape = TensorShape({1, static_cast<int32>(fval_list.size())});

    // todo run embedding here
    
    sparse::SparseTensor val_sparse_tensor(val_indice_tensor, val_list_tensor, val_tensor_shape);
    inputs.push_back(std::pair<std::string, sparse::SparseTensor>("sparse_id_in_field_"+std::to_string(fieldid), id_sparse_tensor));
    inputs.push_back(std::pair<std::string, sparse::SparseTensor>("sparse_val_in_field_"+std::to_string(fieldid), val_sparse_tensor));
  }
  for (int i = 0; i < linear_field.size(); i++) {
    uint32 fieldid = linear_field[i];
    std::vector<int32> indice;
    std::vector<int32> fid_list;
    std::vector<float> fval_list;
    if (instance.find(fieldid) != instance.end()) {
      int num = 0;
      for (std::unordered_map<int32, float>::const_iterator iter = instance[fieldid].begin();
          iter != instance[fieldid].end(); iter++) {
        indice.push_back(0);
        indice.push_back(num++);
        fid_list.push_back(iter->first);
        fval_list.push_back(iter->second);
      }
    } else {
      fid_list.push_back(0); // missid
      fval_list.push_back(0.0);
    }
    auto id_indice_tensor =
      test::AsTensor<int32>(indice, {static_cast<int32>(indice.size()/2), 2});
    auto id_list_tensor = test::AsTensor<int32>(fid_list);
    auto id_tensor_shape = TensorShape({1, static_cast<int32>(fid_list.size())});
    sparse::SparseTensor id_sparse_tensor(id_indice_tensor, id_list_tensor, id_tensor_shape);
    auto val_indice_tensor =
      test::AsTensor<int32>(indice, {static_cast<int32>(indice.size()/2), 2});
    auto val_list_tensor = test::AsTensor<float>(fval_list);
    auto val_tensor_shape = TensorShape({1, static_cast<int32>(fval_list.size())});
    sparse::SparseTensor val_sparse_tensor(val_indice_tensor, val_list_tensor, val_tensor_shape);
    inputs.push_back(std::pair<std::string, sparse::SparseTensor>("linear_id_in_field_"+std::to_string(fieldid), id_sparse_tensor));
    inputs.push_back(std::pair<std::string, sparse::SparseTensor>("linear_val_in_field_"+std::to_string(fieldid), val_sparse_tensor));
  }

  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  // Run the session, evaluating our "softmax" operation from the graph
  //status = session->Run(inputs, {"Softmax"}, {}, &outputs);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    return 1;
  } else {
    std::cout << "Run session successfully" << std::endl;
  }

  // Grab the first output (we only evaluated one graph node: "softmax")
  // and convert the node to a scalar representation.
  auto output_softmax = outputs[0].scalar<float>();

  // (There are similar methods for vectors and matrices here:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

  // Print the results
  std::cout << outputs[0].DebugString() << std::endl;
  std::cout << "output value: " << output_softmax() << std::endl;

  // Free any resources used by the session
  session->Close();
  return 0;
}