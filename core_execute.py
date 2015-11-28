#!/usr/bin/env python
#-*- coding:utf-8 -*-

import pyximport
pyximport.install()
import core_nlp_parse

if __name__ == '__main__':
    core_nlp_parse.parse_method() 
