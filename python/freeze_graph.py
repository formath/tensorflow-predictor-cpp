#!/usr/bin/env python

import sys
import os
import argparse
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

dir = os.path.dirname(os.path.realpath(__file__))

def freeze_graph(checkpoint_dir, graph_pb, output_node_names, output_dir):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        checkpoint_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(checkpoint_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % checkpoint_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_checkpoint_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = output_dir + "/frozen_model.pb"
    output_graph_text = output_dir + "/frozen_model.text"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # import the meta graph and restore checkpoint
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
        saver.restore(sess, input_checkpoint)

        graph = tf.get_default_graph().as_graph_def()
        if graph_pb:
            new_graph = graph_pb2.GraphDef()
            with tf.gfile.FastGFile(graph_pb, 'rb') as f:
                new_graph.ParseFromString(f.read())
            graph = new_graph

        print('<<< Op List Begin <<<')
        for node in graph.node:
            print(node.name)
        print('<<< Op List End <<<')

        # use a built-in TF helper to export variables to constants
        output_node_names = 'init_all_tables,' + output_node_names
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            graph, # The graph_def is used to retrieve the nodes
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        tf.train.write_graph(output_graph_def, output_dir, 'freeze_graph.pb', as_text=False)
        tf.train.write_graph(output_graph_def, output_dir, 'freeze_graph.txt', as_text=True)
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--graph_pb", type=str, default="")
    parser.add_argument("--output_node_names", type=str, default="", help="The name of the output nodes, comma separated.")
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    freeze_graph(args.checkpoint_dir, args.graph_pb, args.output_node_names, args.output_dir)