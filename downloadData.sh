#!/bin/bash

mkdir -p data && rm -rf data/*

cd data && curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2895/ud-treebanks-v2.3.tgz

for file in `ls *.tgz`;
do
  tar -xzf $file
  rm $file
done

cd ud-treebanks-v2.3

#Remove unmergable or merg datasets
#Remove UD_Arabic-NYUAD - requires licences
rm -rf UD_Arabic-NYUAD

# merge English-ESL
cd UD_English-ESL/
curl --remote-name-all https://s3-eu-west-1.amazonaws.com/ilexir-website-media/fce-released-dataset.zip
unzip fce-released-dataset.zip && rm fce-released-dataset.zip
python2 merge.py
rm *.conllu
unzip data.zip && rm data.zip
mv data/corrected/*.conllu ./
rm -rf data fce-released-dataset
cd ..

# Remove UD_Hindi_English-HIENCS - requires twitter API
rm -rf UD_Hindi_English-HIENCS
# Remove UD_Hindi_English-HIENCS - requires log in to download core dataset
rm -rf UD_Japanese-BCCWJ