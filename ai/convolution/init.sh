#!/bin/bash

if [[ "$1" == "docker" ]]; then
  docker run --gpus all --privileged --rm -it -v $PWD:/workspace pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
elif [[ "$1" == "build" ]]; then
  python setup.py install
elif [[ "$1" == "clean" ]]; then
  python setup.py clean
  rm -rf build dist *.egg*
elif [[ "$1" == "run" ]]; then
  python run_test.py
elif [[ "$1" == "profile" ]]; then
  nvprof  --metrics dram_utilization,warp_execution_efficiency,sm_efficiency,achieved_occupancy python run_test.py
else
  python run_test.py
fi
