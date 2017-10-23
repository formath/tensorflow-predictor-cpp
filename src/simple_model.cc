#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace tensorflow;

/**
 * @brief deep model for click through rate prediction
 * @details [long description]
 *
 * @param argv[1] graph protobuf
 *
 * @return [description]
 */
int main(int argc, char* argv[]) {
  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return 1;
  } else {
    std::cout << "Session created successfully" << std::endl;
  }

  // Load the protobuf graph
  GraphDef graph_def;
  std::string graph_path = argv[1];
  status = ReadBinaryProto(Env::Default(), graph_path, &graph_def);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return 1;
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
    std::cerr << status.ToString() << std::endl;
    return 1;
  } else {
    std::cout << "Run session successfully" << std::endl;
  }

  // Grab the first output (we only evaluated one graph node: "c")
  // and convert the node to a scalar representation.
  auto output_c = outputs[0].scalar<float>();

  // (There are similar methods for vectors and matrices here:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

  // Print the results
  std::cout << outputs[0].DebugString() << std::endl; // Tensor<type: float shape: [] values: 30>
  std::cout << "output value: " << output_c() << std::endl; // 30

  // Free any resources used by the session
  session->Close();

  return 0;
}