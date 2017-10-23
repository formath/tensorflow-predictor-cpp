#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
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
#include "dict.pb.h"

using namespace tensorflow;

/**
 * @brief deep model for click through rate prediction
 * @details [long description]
 *
 * @param argv[1] sparse fieldid list seperated by ','
 * @param argv[2] field featureid dict protobuf
 * @param argv[3] freeze graph protobuf
 *
 * @return [description]
 */
int main(int argc, char* argv[]) {
  // parse field
  std::vector<std::string> tokens;
  std::vector<int> sparse_field; // sparse field
  util::split(argv[1], ',', tokens);
  for (std::string token: tokens) {
    sparse_field.push_back(std::stoi(token));
  }

  // load dict protobuf
  ::protobuf::Dict dict;
  std::fstream input(argv[2], std::ios::in | std::ios::binary);
  if (!dict.ParseFromIstream(&input)) {
    std::cerr << "Failed to load dict protobuf" << std::endl;
    return 1;
  }
  std::cout << "load feature num: " << dict.featureid2sortid_size() << std::endl;
  for (::google::protobuf::Map<::google::protobuf::uint32, ::google::protobuf::uint64>::const_iterator iter = dict.field2missid().begin();
    iter != dict.field2missid().end(); iter++) {
    std::cout << "fieldid: " << iter->first << " missid: " << iter->second << std::endl;
  }
  for (::google::protobuf::Map<::google::protobuf::uint32, ::google::protobuf::uint64>::const_iterator iter = dict.field2feanum().begin();
    iter != dict.field2feanum().end(); iter++) {
    std::cout << "fieldid: " << iter->first << " feanum: " << iter->second << std::endl;
  }

  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cerr << status.ToString() << "\n";
    return 1;
  } else {
    std::cout << "Session created successfully" << std::endl;
  }

  // Read in the protobuf graph we exported
  // (The path seems to be relative to the cwd. Keep this in mind
  // when using `bazel run` since the cwd isn't where you call
  // `bazel run` but from inside a temp folder.)
  GraphDef graph_def;
  std::string graph_path = argv[3];
  status = ReadBinaryProto(Env::Default(), graph_path, &graph_def);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
  } else {
    std::cout << "Load graph protobuf successfully" << std::endl;
  }

  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return 1;
  } else {
    std::cout << "Add graph to session successfully" << std::endl;
  }

  // Setup inputs and outputs
  // input 9:283:1 6:384:1 152:384:1
  std::string libfm_data = "9:283:1 6:384:1 152:384:1";
  std::unordered_map<int64, std::unordered_map<int64, float> > instance;
  std::vector<std::string> features;
  util::split(libfm_data, ' ', features);
  for (std::string feature: features) {
    std::vector<std::string> tokens;
    util::split(feature, ':', tokens);
    int64 fieldid = std::stoi(tokens[0]);
    int64 featureid std::stoi(tokens[1]);
    float value = std::stof(tokens[2]);
    if (instance.find(fieldid) == instance.end()) {
      std::unordered_map<int64, float> f;
      f[featureid] = value;
      instance[fieldid] = f;
    } else {
      instance[fieldid][featureid] = value;
    }
  }

  // tensor inputs
  std::vector<std::pair<std::string, Tensor> > inputs;

  for (int i = 0; i < sparse_field.size(); i++) {
    uint64 fieldid = sparse_field[i];
    std::vector<int64> indice;
    std::vector<int64> fid_list;
    std::vector<float> fval_list;
    if (instance.find(fieldid) != instance.end()) {
      int num = 0;
      for (std::unordered_map<int64, float>::const_iterator iter = instance[fieldid].begin();
          iter != instance[fieldid].end(); iter++) {
        indice.push_back(0);
        indice.push_back(num++);
        if (dict.featureid2sortid().find(iter->first) != dict.featureid2sortid().end()) {
          fid_list.push_back(dict.featureid2sortid().find(iter->first)->second);
          fval_list.push_back(iter->second);
        } else {
          fid_list.push_back(dict.field2missid().find(fieldid)->second);
          fval_list.push_back(0);
        }
      }
    } else {
      fid_list.push_back(dict.field2missid().find(fieldid)->second);
      fval_list.push_back(0);
    }

    // input/sparse_id/index/Placeholder
    auto id_indice_tensor =
      test::AsTensor<int64>(indice, {static_cast<int64>(indice.size()/2), 2});
    inputs.push_back(std::pair<std::string, Tensor>("input/sparse_" + std::to_string(fieldid) +"/index/Placeholder", id_indice_tensor));

    // input/sparse_id/id/Placeholder
    auto id_list_tensor = test::AsTensor<int64>(fid_list);
    inputs.push_back(std::pair<std::string, Tensor>("input/sparse_" + std::to_string(fieldid) +"/id/Placeholder", id_list_tensor));

    // input/sparse_id/shape/Placeholder not used. Why?
    auto id_tensor_shape = TensorShape({1, static_cast<int64>(fid_list.size())});

    // input/sparse_id/value/Placeholder
    auto val_list_tensor = test::AsTensor<float>(fval_list);
    inputs.push_back(std::pair<std::string, Tensor>("input/sparse_" + std::to_string(fieldid) +"/value/Placeholder", val_list_tensor));
  }

  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  // Run the session, evaluating our "logit" operation from the graph
  status = session->Run(inputs, {"predict/add"}, {}, &outputs);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return 1;
  } else {
    std::cout << "Run session successfully" << std::endl;
  }

  // Grab the first output (we only evaluated one graph node: "logit")
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