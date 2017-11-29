#!/bin/bash

# Downloads and uncompresses all the required D1 files.
wget -O D1.zip https://www.dropbox.com/s/x86dqgk96sjgbyl/D1.zip?dl=1
unzip -o D1.zip
rm D1.zip
echo All required D1 files have been downloaded and un-compressed successfully.

# Downloads and uncompresses all the required D2 files.
wget -O D2.zip https://www.dropbox.com/s/dll9z2sb1pfbl6c/D2.zip?dl=1
unzip -o D2.zip
rm D2.zip
echo All required D2 files have been downloaded and un-compressed successfully.
