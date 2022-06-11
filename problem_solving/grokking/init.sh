#!/bin/bash

ARG01=$1

if [[ "$ARG01" == "docker" ]]; then
  docker run --gpus all --privileged --rm -it -v $PWD:/workspace pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
elif [[ "$ARG01" == "build" ]]; then
  $CC -o bin.$ARG01 $ARG01 $FLAGS $DEBUG
elif [[ "$ARG01" == "clean" ]]; then
  rm -rf bin.*
else
  if [ ! -f "$ARG01" ]; then
    echo "The file \"$ARG01\" does not exist!"
    echo "Execute: ./init.sh FilePath"
  else
    filename=$(basename $ARG01)
    FLAGS='--std=c++11'
    DEBUG='-Wall -O0 -g'
    CC=clang++
    rm -rf bin.*
    $CC -o bin.$filename $ARG01 $FLAGS $DEBUG && ./bin.$filename
    rm -rf bin.*
  fi
fi
