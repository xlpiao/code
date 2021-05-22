#!/bin/bash

if [[ "$1" == "docker" ]]; then
  docker run --gpus all --privileged --rm -it -v $PWD:/workspace pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
elif [[ "$1" == "build" ]]; then
  $CC -o bin.$1 $1 $FLAGS $DEBUG
elif [[ "$1" == "clean" ]]; then
  rm -rf bin.*
else
  if [ ! -f "$1" ]; then
    echo "The file \"$1\" does not exist!"
    echo "Execute: ./init.sh FileName"
  else
    FLAGS='--std=c++11'
    DEBUG='-Wall -O0 -g'
    CC=clang++
    rm -rf bin.*
    $CC -o bin.$1 $1 $FLAGS $DEBUG
    ./bin.$1
    rm -rf bin.*
  fi
fi
