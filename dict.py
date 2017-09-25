#!/usr/bin/env python

import sys
import os
import pickle

class Dict:
	def __init__(self, continuous_fields, sparse_fields, linear_fields):
		self.ParseFields(continuous_fields, sparse_fields, linear_fields)

	# three type of field: continuous, sparse and linear
    # sparse and linear can have the same one fieldid
	def ParseFields(self, continuous_fields, sparse_fields, linear_fields):
		self.continuous_fields = [int(x) for x in continuous_fields.split(',')]
		self.sparse_fields = [int(x) for x in sparse_fields.split(',')]
		self.linear_fields = [int(x) for x in linear_fields.split(',')]
		print('continuous field: ' + continuous_fields)
		print('sparse field: ' + sparse_fields)
		print('linear field: ' + linear_fields)

    # generate fieldid and its featureid dict
    # field : {featureid : sortid, featureid : sortid, 'miss' : sortid, 'num': feature_num}
	def Parse(self, input_file, output_file):
		field_feature_dict = {}
		field_sortid = {}
		print('Start to generate dict from file ' + input_file)
		for line in open(input_file, 'r'):
			line = line.rstrip('\t\n ')
			tokens = line.split(' ')
			for token in tokens[1:]:
				fieldid, featureid, value = token.split(':')
				if int(fieldid) not in self.sparse_fields and  int(fieldid) not in self.linear_fields:
					continue
				if int(fieldid) not in field_sortid:
					field_sortid[int(fieldid)] = 0
					feature_sortid = {}
					feature_sortid[int(featureid)] = field_sortid[int(fieldid)]
					field_sortid[int(fieldid)] += 1
					field_feature_dict[int(fieldid)] = feature_sortid
				else:
					if int(featureid) not in field_feature_dict[int(fieldid)]:
						field_feature_dict[int(fieldid)][int(featureid)] = field_sortid[int(fieldid)]
						field_sortid[int(fieldid)] += 1
		for fieldid in field_feature_dict:
			field_feature_dict[fieldid]['miss'] = field_sortid[fieldid]
			field_feature_dict[fieldid]['num'] = field_sortid[fieldid] + 1

		print('field num: ' + str(len(field_feature_dict)))
		for fieldid in field_feature_dict:
			print('field: ' + str(fieldid) + ' feature num: ' + str(field_feature_dict[fieldid]['num']))

		output = open(output_file, 'wb')
		pickle.dump(field_feature_dict, output, 2)
		print('Successfully generate dict from {} to {}'.format(input_file, output_file))

if __name__ == '__main__':
	if len(sys.argv) != 6:
		print('''
			Usage: python dict.py continuous_fields sparse_fields linear_fields input_file output_file
			params: continuous_fields, example 0,1,3,4
					sparse_fields, example 495, 38, 24
					linear_fields, those fields are also sparse, example 37,28,23
					input_file, libfm data
					output_file, dict file to generate
			''')
		exit(1)
	dict = Dict(sys.argv[1], sys.argv[2], sys.argv[3])
	dict.Parse(sys.argv[4], sys.argv[5])
