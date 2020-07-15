#!/bin/bash
# download wget with homebrew

# we can change it to run setup.py directly from here.

PROJECT_ID=postal-parser

brew install wget
mkdir temp
cd temp
wget https://raw.githubusercontent.com/HSBC-Internship-2020/postal-parser/addressformatter/data/test_CoNLL_addresses.txt



gsutil cp test_CoNLL_addresses.txt gs://${PROJECT_ID}/pyspark_nlp/data/test_CoNLL_addresses.txt
rm test_CoNLL_addresses.txt

rm -rf temp


# have a beautiful day :)

