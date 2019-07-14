#!/bin/bash

mkdir -p data && cd data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B7OEHD3T4eCkVGs0TkhUWFN6N1k' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B7OEHD3T4eCkVGs0TkhUWFN6N1k" -O 300W-LP.zip
unzip 300W-LP.zip
