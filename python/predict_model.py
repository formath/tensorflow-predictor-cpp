import sys
import os
import datetime
import tensorflow as tf
from deep_model import Model
from data import Data

# This model is the same with deep_model
# except the input part replaced by placeholder for feed Tensor on prediction.
# The origin model during training have tf.Example node in graph
# which is not approriate for online prediction because can't feed Tensor.

# config
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dict', './libfm.dict', 'field feature dict')
flags.DEFINE_string("model_dir", "./model", "model dirctory")
flags.DEFINE_string('sparse_fields', '', 'sparse fields. example 0,1,2')
flags.DEFINE_string('hidden_layer', '100,100,50', 'hidden size for eacy layer')
flags.DEFINE_integer('embedding_size', 10, 'embedding size')


# data iter
data = Data(FLAGS.dict, FLAGS.sparse_fields)
label, sparse_id, sparse_val = data.ReadBatchPlaceholder()

# define model
model = Model(FLAGS.embedding_size, data.Dict(), FLAGS.sparse_fields, FLAGS.hidden_layer)

# define loss
logits, all_parameter = model.forward(sparse_id, sparse_val)
train_label = tf.to_int64(label)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=train_label, name='cross_entropy')

# save graph
sess = tf.Session()
tf.train.write_graph(sess.graph.as_graph_def(), FLAGS.model_dir, 'predict_graph.pb', as_text=False)
tf.train.write_graph(sess.graph.as_graph_def(), FLAGS.model_dir, 'predict_graph.txt', as_text=True)
