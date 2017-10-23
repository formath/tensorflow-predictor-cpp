#!/usr/bin/env python

import sys
import os
import tensorflow as tf

class Model:
    def __init__(self, embedding_size, field_feature_dict, sparse_field, hidden_layer):
        self.embedding_size = embedding_size
        self.field_feature_dict = field_feature_dict
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
            input_size = self.field_feature_dict.field2feanum[field_id] + 1
            with tf.variable_scope("emb_"+str(field_id)):
                embedding_variable = tf.Variable(tf.truncated_normal([input_size, self.embedding_size], stddev=0.05), name='emb' + str(field_id))
                embedding = tf.nn.embedding_lookup_sparse(embedding_variable, sparse_ids[i], sparse_vals[i], "mod", combiner="sum")
                emb.append(embedding)
            self.embedding.append(embedding_variable)

        return tf.concat(emb, 1, name='concat_embedding')

    def forward(self, sparse_id, sparse_val):
        '''
        forward graph
        '''

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
            with tf.variable_scope("hidden"):
                weight = tf.Variable(tf.truncated_normal([dim, hidden_size], stddev=0.05), name='fully_weight_'+str(i))
                bias = tf.Variable(tf.truncated_normal([hidden_size], stddev=0.05), name='fully_bias_'+str(i))
            self.hiddenW.append(weight)
            self.hiddenB.append(bias)
            net = tf.nn.relu(tf.matmul(net, weight) + bias, name='fully_'+str(i))

        #dim = net.get_shape().as_list()[1]
        dim = self.hidden_layer[-1]
        print("out layer dim:" + str(dim))
        with tf.variable_scope("outlayer"):
            self.weight = tf.Variable(tf.truncated_normal([dim, 1], stddev=0.05), name='weight_out')
            self.bias = tf.Variable(tf.truncated_normal([1], stddev=0.05), name='bias_out')
        with tf.variable_scope("predict"):
            logits = tf.matmul(net, self.weight) + self.bias

        # add regularization
        all_parameter = [self.weight, self.bias] + self.hiddenW + self.hiddenB + self.embedding

        return logits, all_parameter
