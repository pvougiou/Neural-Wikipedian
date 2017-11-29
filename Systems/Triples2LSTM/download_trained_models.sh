#!/bin/bash

# Downloads and uncompresses our trained Triples2LSTM models on both D1 and D2 with
# URIs and surface form tuples.
wget -O checkpoints.zip https://www.dropbox.com/s/vpam8zln4xnrpqd/checkpoints.zip?dl=1
unzip -o checkpoints.zip 
rm checkpoints.zip
echo All required checkpoint files have been downloaded and un-compressed successfully.
