#!/bin/bash

## global variables
PROJECT_DIR=$(realpath $(dirname ${0}))
echo ${PROJECT_DIR}

## ansi colors
REDBOLD="\033[1;31m"
GREENBOLD="\033[1;32m"
PURPLEBOLD="\033[1;34m"
WHITEBOLD="\033[1;37m"
RESET="\033[0m"

## functions
function check() {
  ERROR=${?}
  if [[ ${ERROR} == 0 ]]; then
    printf "${GREENBOLD}%s Success !!!${RESET}\n\n" "${1}"
  else
    printf "${REDBOLD}%s Failed With Error Code ${ERROR} !!!${RESET}\n\n" "${1}"
    exit ${ERROR}
  fi
}

## main
FLAGS='--std=c++17'
DEBUG='-Wall -O0 -g'
CC=clang++
if [[ "${1}" == "docker" ]]; then
  docker run --gpus all --privileged --rm -it -v ${PROJECT_DIR}:/work pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
elif [[ "${1}" == "build" ]]; then
  $CC -o bin.${1} ${1} ${FLAGS} ${DEBUG}
elif [[ "${1}" == "clean" ]]; then
  rm -rf bin.*
else
  if [ ! -f "${1}" ]; then
    echo "The file \"${1}\" does not exist!"
    echo "Execute: ./init.sh FileName"
  else
    rm -rf bin.*
    $CC -o bin.${1} ${1} ${FLAGS} ${DEBUG}; check "Compile"
    ./bin.${1}; check "Run"; rm -rf bin.*
  fi
fi
