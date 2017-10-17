import sys
import os
import argparse
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--graph_pb", type=str, default="")
    parser.add_argument("--output_node_names", type=str, default="", help="The name of the output nodes, comma separated.")
    parser.add_argument("--output_pb", type=str, default="")
    args = parser.parse_args()

    freeze_graph.freeze_graph(args.graph_pb, '', True,
               args.checkpoint_dir, args.output_node_names,
               '', '',
               args.output_pb, True, '')