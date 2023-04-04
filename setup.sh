#!/bin/bash
# File              : setup.sh
# Author            : Xianglan Piao <lanxlpiao@gmail.com>
# Date              : 2020.04.30
# Last Modified Date: 2020.05.21
# Last Modified By  : Xianglan Piao <lanxlpiao@gmail.com>

file=.clang-format

clang-format -style=google -dump-config > ${file}

if [ "$(uname)" = "Linux" ]; then
  sed -i '/^$/d' ${file}
elif [ "$(uname)" = "Darwin" ]; then
  sed -i '' -e '$d' ${file}
else
  sed -i '/^$/d' ${file}
fi

git add ${file}
git config --local core.hooksPath $PWD/hooks
