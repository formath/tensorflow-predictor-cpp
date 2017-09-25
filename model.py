#!/usr/bin/env python

import sys
import os
import tensorflow as tf

class Model:
    def __init__(self, embedding_size, field_feature_dict, sparse_field, continuous_field, linear_field, hidden_layer):
        self.embedding_size = embedding_size
        self.field_feature_dict = field_feature_dict
        self.sparse_field =[]
        if sparse_field != '':
            for i in sparse_field.split(','):
                self.sparse_field.append(int(i))
        self.continuous_field = []
        if continuous_field != '':
            for i in continuous_field.split(','):
                self.continuous_field.append(int(i))
        self.linear_field = []
        if linear_field != '':
            for i in linear_field.split(','):
                self.linear_field.append(int(i))
        self.hidden_layer = []
        for i in hidden_layer.split(','):
            self.hidden_layer.append(int(i))

    # sparse embedding and concat all field embedding
    def concat(self, fields, sparse_id, sparse_val):
        emb = []
        for i, field_id in enumerate(fields):
            input_size = self.field_feature_dict[field_id]['num'] + 1
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

        net = self.concat(self.sparse_field, sparse_id, sparse_val)
        if len(self.continuous_field) > 0:
            net = tf.concat([net, continuous_val], 1, name='concat_sparse_continuous')

        #hidden layers
        for i, hidden_size in enumerate(self.hidden_layer):
            dim = net.get_shape().as_list()[1]
            with tf.variable_scope("hidden") as scope:
                weight = tf.Variable(tf.truncated_normal([dim, hidden_size], stddev=0.05), name='fully_weight_'+str(i))
                bias = tf.Variable(tf.truncated_normal([hidden_size], stddev=0.05), name='fully_bias_'+str(i))
            self.hiddenW.append(weight)
            self.hiddenB.append(bias)
            net = tf.nn.relu(tf.matmul(net, weight) + bias, name='fully_'+str(i))
            #net = tf.nn.dropout(net, self.drop_out, name='dropout_'+str(i))
            #tf.summary.histogram('hidden_w' + str(i), weight)

        # merge linear sparse
        if len(self.linear_field) > 0:
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
