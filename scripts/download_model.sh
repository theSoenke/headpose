#!/bin/bash

mkdir -p data && cd data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR" -O hopenet_robust_alpha1.pkl
