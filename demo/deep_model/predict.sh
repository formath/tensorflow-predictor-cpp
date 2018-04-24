#!/usr/bin/env bash

export TF_CPP_MIN_LOG_LEVEL=3
../../bin/deep_model.bin \
	"9,6,116,152,179" \
	"./model/freeze_graph.pb"