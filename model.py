#!/usr/bin/env python

import sys
import os
import tensorflow as tf

class Model:
    def __init__(self, embedding_size, field_feature_dict, sparse_field, continuous_field, linear_field, model_dir, hidden_layer, algo='adam', drop_out=1.0, alpha=0.0, beta=0.0, learning_rate=0.01):
        self.embedding_size = embedding_size
        self.field_feature_dict = field_feature_dict
        self.sparse_field = sparse_field
        self.continuous_field = continuous_field
        self.linear_field = linear_field
        self.model_dir = model_dir
        self.alpha = alpha
        self.beta = beta
        self.drop_out = drop_out
        self.learning_rate = learning_rate
        self.algo = algo
        self.hidden_layer = hidden_layer.split(",")

    # sparse embedding and concat all field embedding
    def concat(self, fields, sparse_id, sparse_val):
        emb = []
        for i, field_id in enumerate(fields):
            input_size = self.field_feature_dic[field_id]['index'] + 1
            with tf.variable_scope("emb") as scope:
                embedding_variable = tf.Variable(tf.truncated_normal([input_size, self.embedding_size], stddev=0.05), name='emb' + str(field_id))
            embedding = tf.nn.embedding_lookup_sparse(embedding_variable, sparse_id[i], sparse_val[i], "mod", combiner="sum")
            emb.append(embedding)
            #tf.summary.histogram('emb_' + str(field_id), embedding_variable)
            self.embedding.append(embedding_variable)

        return tf.concat(emb, 1, name='concat_embedding')

    def forward(self, sparse_id, sparse_val, linear_id, linear_val, continuous_val):
        '''
        forward graph
        '''

        self.embedding = []
        self.hiddenW = []
        self.hiddenB = []

        sparse_embedding = self.concat(self.sparse_field, sparse_id, sparse_val)
        net = tf.concat([sparse_embedding, continuous_val], 1, name='concat_sparse_continuous')

        #hidden layers
        for i, hidden_size in enumerate(self.hidden_layer):
            dim = net.get_shape().as_list()[1]
            with tf.variable_scope("hidden") as scope:
                weight = tf.Variable(tf.truncated_normal([dim, int(hidden_size)], stddev=0.05), name='fully_weight_'+str(i))
                bias = tf.Variable(tf.truncated_normal([int(hidden_size)], stddev=0.05), name='fully_bias_'+str(i))
            self.hiddenW.append(weight)
            self.hiddenB.append(bias)
            net = tf.nn.relu(tf.matmul(net, weight) + bias, name='fully_'+str(i))
            #net = tf.nn.dropout(net, self.drop_out, name='dropout_'+str(i))
            #tf.summary.histogram('hidden_w' + str(i), weight)

        # merge linear sparse
        if linear_placeholder is not None:
            linear_embedding = self.concat(self.linear_field, linear_id, linear_val)
            net = tf.concat([net, linear_embedding], 1, name='concat_linear')

        dim = net.get_shape().as_list()[1]
        print("out layer dim:" + str(dim))
        with tf.variable_scope("outlayer") as scope:
            self.weight = tf.Variable(tf.truncated_normal([dim, 1], stddev=0.05), name='weight_out')
            self.bias = tf.Variable(tf.truncated_normal([1], stddev=0.05), name='bias_out')
        logits = tf.matmul(net, self.weight) + self.bias

        # add regularization
        all_parameter = [self.weight, self.bias] + self.hiddenW + self.hiddenB + self.embedding

        return logits, all_parameter
