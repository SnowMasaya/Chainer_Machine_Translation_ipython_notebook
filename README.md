Chainer with Machine Translation Tech-Circle#11 
====

This tool is making the Neural Networks Language Model

>[Japanese Reference Pages]<br>


## Description
This tool is making the Machine Translation Model

If you see the detail about it, you see the below<br> 
#
### Install

If you don't install pyenv and virtualenv you have to install bellow
####Prepare Install
linux
```
apt-get install pyenv 
apt-get install virtualenv 
```
Mac
```
brew install pyenv 
brew install virtualenv 
```

####Prepare Inastall2
```
pyenv install 3.4.1
pyenv rehash
pyenv local 3.4.1
virtualenv -p ~/.pyenv/versions/3.4.1/bin/python3.4 my_env
source my_env/bin/activate

```

```
pip install -r requirement.txt 
```
Installing a library bellow
##Requirements

    Python 3.4+
    numpy==1.9.2
    chainer==1.4.0
    ipython==4.0.0
    notebook==4.0.4
    jinja2==2.8
    pyzmq==14.7.0
    tornado==4.1
    corenlp-python

####Confirm library

```
ipython
```

Type command bellow
```
import math
import sys
import time

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
```

#
### Usage 
#
```
*You execute python 
ipython notebook
```
#
### Data Directory Structure 
#
```
samples/　　　　　... Sample model and Translation result
  - middle/ 　　　　　... middle setteing
train/　     　... training data set
test/ 　　　　　　　... test data set
```
#
### Licence
#
```
The MIT License (MIT)

Copyright (c) 2015 Masaya Ogushi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
#
### Author
#
[SnowMasaya](https://github.com/SnowMasaya)
### References 
#
>[Chainer]http://chainer.org/<br>
>[Chainer Machine Translation]https://github.com/odashi/chainer_examples<br>
>[Learning and Predict Data]http://www2.nict.go.jp/univ-com/multi_trans/member/mutiyama/manual/index-ja.html<br>

