#!/usr/bin/env python
#-*- coding:utf-8 -*-

import pprint
import json
import corenlp

# パーサの生成
corenlp_dir = "/usr/local/lib/stanford-corenlp-full-2013-06-20/"
properties_file = "./user.properties"
parser = corenlp.StanfordCoreNLP(
    corenlp_path=corenlp_dir,
    properties=properties_file) # propertiesを設定

target_file = "target_tmp.txt"

def parse_method():
# パースして結果をpretty print
    for line in open(target_file, 'r'):
        result_json = json.loads(parser.parse(line.strip()))
        words = result_json["sentences"][0]["words"]
        for i in range(len(words)):
            print str(words[i][0]) + " ", 
            if i == len(words) - 1:
                print("")
