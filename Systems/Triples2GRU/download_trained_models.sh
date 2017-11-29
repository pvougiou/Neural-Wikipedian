#!/bin/bash

# Downloads and uncompresses our trained Triples2GRU models on both D1 and D2 with
# URIs and surface form tuples.
wget -O checkpoints.zip https://www.dropbox.com/s/l8uf5q0q5d4kg29/checkpoints.zip?dl=1
unzip -o checkpoints.zip
rm checkpoints.zip
echo All required checkpoint files have been downloaded and un-compressed successfully.
