#!/bin/bash

rm -rf ./data/*

cd data && curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2895/ud-treebanks-v2.3.tgz

for file in `ls *.tgz`;
do
  tar -xzf $file
  rm $file
done
