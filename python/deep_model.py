#!/usr/bin/env python

import sys
import os
import tensorflow as tf

class Model:
    def __init__(self, embedding_size, sparse_field, hidden_layer):
        self.embedding_size = embedding_size
        self.sparse_field =[]
        if sparse_field != '':
            for i in sparse_field.split(','):
                self.sparse_field.append(int(i))
        self.hidden_layer = []
        for i in hidden_layer.split(','):
            self.hidden_layer.append(int(i))

    # sparse embedding and concat all field embedding
    def concat(self, fields, sparse_ids, sparse_vals):
        emb = []
        for i, field_id in enumerate(fields):
            mapping_ints = tf.constant([0])
            table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_ints, num_oov_buckets=100000, dtype=tf.int64)
            sparse_id_in_this_field = table.lookup(sparse_ids[i])
            with tf.variable_scope("emb_"+str(field_id)):
                embedding_variable = tf.Variable(tf.truncated_normal([100002, self.embedding_size], stddev=0.1))
                embedding = tf.nn.embedding_lookup_sparse(embedding_variable, sparse_id_in_this_field, sparse_vals[i], "mod", combiner="sum")
                emb.append(embedding)
            self.embedding.append(embedding_variable)

        return tf.concat(emb, 1, name='concat_embedding')

    def forward(self, sparse_id, sparse_val):
        '''
        forward graph
        '''

        with tf.variable_scope("forward"):
            self.embedding = []
            self.hiddenW = []
            self.hiddenB = []

            # sparse field embedding
            net = self.concat(self.sparse_field, sparse_id, sparse_val)

            #hidden layers
            for i, hidden_size in enumerate(self.hidden_layer):
                #dim = net.get_shape().as_list()[1]
                if i == 0:
                    dim = self.embedding_size * len(self.sparse_field)
                else:
                    dim = self.hidden_layer[i-1]
                weight = tf.Variable(tf.truncated_normal([dim, hidden_size], stddev=0.1), name='fully_weight_'+str(i))
                bias = tf.Variable(tf.truncated_normal([hidden_size], stddev=0.1), name='fully_bias_'+str(i))
                self.hiddenW.append(weight)
                self.hiddenB.append(bias)
                net = tf.nn.relu(tf.matmul(net, weight) + bias, name='fully_'+str(i))

            #dim = net.get_shape().as_list()[1]
            dim = self.hidden_layer[-1]
            self.weight = tf.Variable(tf.truncated_normal([dim, 2], stddev=0.1), name='weight_out')
            self.bias = tf.Variable(tf.truncated_normal([2], stddev=0.1), name='bias_out')
            with tf.variable_scope("logit"):
                logits = tf.matmul(net, self.weight) + self.bias

        # add regularization
        all_parameter = [self.weight, self.bias] + self.hiddenW + self.hiddenB + self.embedding

        return logits, all_parameter
