#!/usr/bin/env python

import sys
import os
from dict_pb2 import Dict as DictPB
import tensorflow as tf

class Data:
    def __init__(self, dict_file, continuous_fields, sparse_fields, linear_fields):
        self.LoadDict(dict_file)
        self.ParseFields(continuous_fields, sparse_fields, linear_fields)

    # load fieldid and its featureid dict
    def LoadDict(self, dict_file):
        f = open(dict_file, 'rb')
        dict = DictPB()
        dict.ParseFromString(f.read())
        self.field_feature_dict = dict
        f.close()
        print('load field num: ' + str(len(self.field_feature_dict.field2missid)))
        for fieldid, feanum in self.field_feature_dict.field2feanum.items():
            print('field: ' + str(fieldid) + ' feature num: ' + str(feanum))


    def Dict(self):
        assert self.field_feature_dict is not None
        return self.field_feature_dict

    # three type of field: continuous, sparse and linear
    # sparse and linear can have the same one fieldid
    def ParseFields(self, continuous_fields, sparse_fields, linear_fields):
        if continuous_fields != '':
            self.continuous_field = [int(x) for x in continuous_fields.split(',')]
        else:
            self.continuous_field = []
        if sparse_fields != '':
            self.sparse_field = [int(x) for x in sparse_fields.split(',')]
        else:
            self.sparse_field = []
        if linear_fields != '':
            self.linear_field = [int(x) for x in linear_fields.split(',')]
        else:
            self.linear_field = []
        print('continuous field: ' + continuous_fields)
        print('sparse field: ' + sparse_fields)
        print('linear field: ' + linear_fields)

    # parse each line of libfm data into tfrecord
    def StringToRecord(self, input_file, output_file):
        print('Start to convert {} to {}'.format(input_file, output_file))
        writer = tf.python_io.TFRecordWriter(output_file)

        for line in open(input_file, 'r'):
            tokens = line.split(' ')
            label = float(tokens[0])
            field2feature = {}
            for fea in tokens[1:]:
                fieldid, featureid, value = fea.split(':')
                if int(fieldid) not in field2feature:
                    feature2value = {}
                    feature2value[int(featureid)] = float(value)
                    field2feature[int(fieldid)] = feature2value
                else:
                    field2feature[int(fieldid)][int(featureid)] = float(value)

            feature = {}
            feature['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
            for fieldid in self.sparse_field:
                feature_id_list = []
                feature_val_list = []
                if fieldid in field2feature:
                    for featureid in field2feature[fieldid]:
                        value = field2feature[fieldid][featureid]
                        feature_id_list.append(self.field_feature_dict.featureid2sortid[featureid])
                        feature_val_list.append(value)
                else:
                    feature_id_list.append(self.field_feature_dict.field2missid[fieldid])
                    feature_val_list.append(0.0)
                feature['sparse_id_in_field_'+str(fieldid)] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature_id_list))
                feature['sparse_val_in_field_'+str(fieldid)] = tf.train.Feature(float_list=tf.train.FloatList(value=feature_val_list))
            for fieldid in self.linear_field:
                feature_id_list = []
                feature_val_list = []
                if fieldid in field2feature:
                    for featureid in field2feature[fieldid]:
                        value = field2feature[fieldid][featureid]
                        feature_id_list.append(self.field_feature_dict.featureid2sortid[featureid])
                        feature_val_list.append(value)
                else:
                    feature_id_list.append(self.field_feature_dict.field2missid[fieldid])
                    feature_val_list.append(0.0)
                feature['linear_id_in_field_'+str(fieldid)] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature_id_list))
                feature['linear_val_in_field_'+str(fieldid)] = tf.train.Feature(float_list=tf.train.FloatList(value=feature_val_list))
            feature_val_list = []
            for fieldid in self.continuous_field:
                if fieldid in field2feature:
                    assert len(field2feature[fieldid]) == 1
                    for featureid in field2feature[fieldid]:
                        value = field2feature[fieldid][featureid]
                        feature_val_list.append(value)
                else:
                    feature_val_list.append(0.0)
            feature['continuous_val'] = tf.train.Feature(float_list=tf.train.FloatList(value=feature_val_list))
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()
        print('Successfully convert {} to {}'.format(input_file, output_file))

    def Decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        return serialized_example

    def ReadBatch(self, file_name, max_epoch, batch_size, thread_num, min_after_dequeue):
        '''
        Return Tensor and SparseTensor parsed from tfrecord
        '''
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(
                tf.train.match_filenames_once(file_name), num_epochs=max_epoch)
            serialized_example = self.Decode(filename_queue)
            capacity = thread_num * batch_size + min_after_dequeue
            batch_serialized_example = tf.train.shuffle_batch(
                                    [serialized_example],
                                    batch_size=batch_size,
                                    num_threads=thread_num,
                                    capacity=capacity,
                                    min_after_dequeue=min_after_dequeue)
            features = {}
            features['label'] = tf.FixedLenFeature([], tf.float32)
            for fieldid in self.sparse_field:
                features['sparse_id_in_field_'+str(fieldid)] = tf.VarLenFeature(tf.int64)
                features['sparse_val_in_field_'+str(fieldid)] = tf.VarLenFeature(tf.float32)
            for fieldid in self.linear_field:
                features['linear_id_in_field_'+str(fieldid)] = tf.VarLenFeature(tf.int64)
                features['linear_val_in_field_'+str(fieldid)] = tf.VarLenFeature(tf.float32)
            features['continuous_val'] = tf.FixedLenFeature([], tf.float32)
            instance = tf.parse_example(batch_serialized_example, features)

            sparse_id = []
            sparse_val = []
            for fieldid in self.sparse_field:
                sparse_id.append(instance['sparse_id_in_field_'+str(fieldid)])
                sparse_val.append(instance['sparse_val_in_field_'+str(fieldid)])
            linear_id = []
            linear_val = []
            for fieldid in self.linear_field:
                linear_id.append(instance['linear_id_in_field_'+str(fieldid)])
                linear_val.append(instance['linear_val_in_field_'+str(fieldid)])
            continuous_val = instance['continuous_val']
            return instance['label'], sparse_id, sparse_val, linear_id, linear_val, continuous_val

    def ReadBatchPlaceholder(self):
        '''
        Return placeholder
        '''
        with tf.name_scope('input'):
            sparse_id = []
            sparse_val = []
            for fieldid in self.sparse_field:
                with tf.variable_scope('sparse_'+str(fieldid)):
                    with tf.variable_scope('index'):
                        self.sparse_index = tf.placeholder(tf.int64)
                    with tf.variable_scope('id'):
                        self.sparse_ids = tf.placeholder(tf.int64)
                    with tf.variable_scope('value'):
                        self.sparse_vals = tf.placeholder(tf.float32)
                    with tf.variable_scope('shape'):
                        self.sparse_shape = tf.placeholder(tf.int64)
                    sparse_id.append(tf.SparseTensor(self.sparse_index, self.sparse_ids, self.sparse_shape))
                    sparse_val.append(tf.SparseTensor(self.sparse_index, self.sparse_vals, self.sparse_shape))
            linear_id = []
            linear_val = []
            for fieldid in self.linear_field:
                with tf.variable_scope('linear_'+str(fieldid)):
                    with tf.variable_scope('index'):
                        self.linear_index = tf.placeholder(tf.int64)
                    with tf.variable_scope('id'):
                        self.linear_ids = tf.placeholder(tf.int64)
                    with tf.variable_scope('value'):
                        self.linear_vals = tf.placeholder(tf.float32)
                    with tf.variable_scope('shape'):
                        self.linear_shape = tf.placeholder(tf.int64)
                    linear_id.append(tf.SparseTensor(self.linear_index, self.linear_ids, self.linear_shape))
                    linear_val.append(tf.SparseTensor(self.linear_index, self.linear_vals, self.linear_shape))
            with tf.variable_scope('label'):
                self.label = tf.placeholder(tf.float32)
            with tf.variable_scope('continuous'):
                self.continuous_val = tf.placeholder(tf.float32)
            return self.label, sparse_id, sparse_val, linear_id, linear_val, self.continuous_val

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print('''
            Usage: python data.py dict_file continuous_fields sparse_fields linear_fields input_file output_file
            params: dict_file, field feature dict
                    continuous_fields, example 0,1,3,4
                    sparse_fields, example 495, 38, 24
                    linear_fields, those fields are also sparse, example 37,28,23
                    input_file, input libfm data
                    output_file, tfrecord file to be generated
            ''')
        exit(1)
    data = Data(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    data.StringToRecord(sys.argv[5], sys.argv[6])
