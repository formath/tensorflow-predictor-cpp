#!/usr/bin/env python

import sys
import os
from dict_pb2 import Dict as DictPB

class Dict:
	def __init__(self, sparse_fields):
		self.ParseFields(sparse_fields)

	# type of field: sparse
	def ParseFields(self, sparse_fields):
		if sparse_fields != '':
			self.sparse_fields = [int(x) for x in sparse_fields.split(',')]
		print('sparse field: ' + sparse_fields)

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
				if int(fieldid) not in self.sparse_fields:
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

		dict = DictPB()
		for fieldid, feature2sortid_dict in field_feature_dict.items():
			for featureid, sortid in feature2sortid_dict.items():
				if featureid == 'miss':
					dict.field2missid[fieldid] = sortid
					continue
				if featureid == 'num':
					dict.field2feanum[fieldid] = sortid
					continue
				dict.featureid2sortid[featureid] = sortid
		output = open(output_file, 'wb')
		output.write(dict.SerializeToString())
		output.close()
		print('Successfully generate dict from {} to {}'.format(input_file, output_file))

if __name__ == '__main__':
	if len(sys.argv) != 4:
		print('''
			Usage: python dict.py continuous_fields sparse_fields linear_fields input_file output_file
			params: sparse_fields, example "495,38,24"
					input_file, libfm data
					output_file, dict file to generate
			''')
		exit(1)
	dict = Dict(sys.argv[1])
	dict.Parse(sys.argv[2], sys.argv[3])