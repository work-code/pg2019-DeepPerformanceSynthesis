#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_reference_caffenet/solver.prototxt
