#!/usr/bin/env python
#-*- coding:utf-8 -*-
#!/usr/bin/python3

import sys
import math
import numpy as np
from argparse import ArgumentParser

from chainer import functions, optimizers

import util.generators as gens
from util.functions import trace, fill_batch
from util.model_file import ModelFile
from util.vocabulary import Vocabulary

import random
from util.chainer_cpu_wrapper import wrapper
#from util.chainer_gpu_wrapper import wrapper

   
class EncoderDecoderModel:

    def __init__(self, parameter_dict):
        self.parameter_dict       = parameter_dict
        self.source               = parameter_dict["source"]
        self.target               = parameter_dict["target"]
        self.test_source          = parameter_dict["test_source"]
        self.test_target          = parameter_dict["test_target"]
        self.reference_target     = parameter_dict["reference_target"]
        self.vocab                = parameter_dict["vocab"]
        self.embed                = parameter_dict["embed"]
        self.hidden               = parameter_dict["hidden"]
        self.epoch                = parameter_dict["epoch"]
        self.minibatch            = parameter_dict["minibatch"]
        self.generation_limit     = parameter_dict["generation_limit"]

    def make_model(self):
        self.model = wrapper.make_model(
            # encoder
            weight_xi = functions.EmbedID(len(self.src_vocab), self.n_embed),
            weight_ip = functions.Linear(self.n_embed, 4 * self.n_hidden),
            weight_pp = functions.Linear(self.n_hidden, 4 * self.n_hidden),
            # decoder
            weight_pq = functions.Linear(self.n_hidden, 4 * self.n_hidden),
            weight_qj = functions.Linear(self.n_hidden, self.n_embed),
            weight_jy = functions.Linear(self.n_embed, len(self.trg_vocab)),
            weight_yq = functions.EmbedID(len(self.trg_vocab), 4 * self.n_hidden),
            weight_qq = functions.Linear(self.n_hidden, 4 * self.n_hidden),
        )

    def new(self, src_vocab, trg_vocab, n_embed, n_hidden, parameter_dict):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.make_model()
        return self

    def save(self, filename):
        with ModelFile(filename, 'w') as fp:
            self.src_vocab.save(fp.get_file_pointer())
            self.trg_vocab.save(fp.get_file_pointer())
            fp.write(self.n_embed)
            fp.write(self.n_hidden)
            wrapper.begin_model_access(self.model)
            fp.write_embed(self.model.weight_xi)
            fp.write_linear(self.model.weight_ip)
            fp.write_linear(self.model.weight_pp)
            fp.write_linear(self.model.weight_pq)
            fp.write_linear(self.model.weight_qj)
            fp.write_linear(self.model.weight_jy)
            fp.write_embed(self.model.weight_yq)
            fp.write_linear(self.model.weight_qq)
            wrapper.end_model_access(self.model)

    def load(self, filename):
        with ModelFile(filename) as fp:
            self.src_vocab = Vocabulary.load(fp.get_file_pointer())
            self.trg_vocab = Vocabulary.load(fp.get_file_pointer())
            self.n_embed = int(fp.read())
            self.n_hidden = int(fp.read())
            self.make_model()
            wrapper.begin_model_access(self.model)
            fp.read_embed(self.model.weight_xi)
            fp.read_linear(self.model.weight_ip)
            fp.read_linear(self.model.weight_pp)
            fp.read_linear(self.model.weight_pq)
            fp.read_linear(self.model.weight_qj)
            fp.read_linear(self.model.weight_jy)
            fp.read_embed(self.model.weight_yq)
            fp.read_linear(self.model.weight_qq)
            wrapper.end_model_access(self.model)
        return self

    def init_optimizer(self):
        self.__opt = optimizers.AdaGrad(lr=0.01)
        self.__opt.setup(self.model)

    def forward(self, is_training, src_batch, trg_batch = None, generation_limit = None):
        pass

    def train(self, src_batch, trg_batch):
        self.__opt.zero_grads()
        hyp_batch, accum_loss = self.forward(True, src_batch, trg_batch=trg_batch)
        accum_loss.backward()
        self.__opt.clip_grads(10)
        self.__opt.update()
        return hyp_batch

    def predict(self, src_batch, generation_limit):
        return self.forward(False, src_batch, generation_limit=generation_limit)

    def train_model(self):
        trace('making vocaburaries ...')
        src_vocab = Vocabulary.new(gens.word_list(self.source), self.vocab)
        trg_vocab = Vocabulary.new(gens.word_list(self.target), self.vocab)

        trace('making model ...')
        model = self.new(src_vocab, trg_vocab, self.embed, self.hidden, self.parameter_dict)

        random_number = random.randint(0, self.minibatch)
        for i_epoch in range(self.epoch):
            trace('epoch %d/%d: ' % (i_epoch + 1, self.epoch))
            trained = 0
            gen1 = gens.word_list(self.source)
            gen2 = gens.word_list(self.target)
            gen3 = gens.batch(gens.sorted_parallel(gen1, gen2, 100 * self.minibatch), self.minibatch)
            model.init_optimizer()

            for src_batch, trg_batch in gen3:
                src_batch = fill_batch(src_batch)
                trg_batch = fill_batch(trg_batch)
                K = len(src_batch)
                hyp_batch = model.train(src_batch, trg_batch)

                if trained == 0:
                    self.print_out(random_number, i_epoch, trained, src_batch, trg_batch, hyp_batch)

                trained += K

            trace('saving model ...')
            model.save("ChainerMachineTranslation" + '.%03d' % (self.epoch + 1))

        trace('finished.')

    def test_model(self, model_name):
        trace('loading model ...')
        model = self.load(model_name)
    
        trace('generating translation ...')
        generated = 0

        with open(self.test_target, 'w') as fp:
            for src_batch in gens.batch(gens.word_list(self.test_source), self.minibatch):
                src_batch = fill_batch(src_batch)
                K = len(src_batch)

                trace('sample %8d - %8d ...' % (generated + 1, generated + K))
                hyp_batch = model.predict(src_batch, self.generation_limit)

                source_cuont = 0
                for hyp in hyp_batch:
                    hyp.append('</s>')
                    hyp = hyp[:hyp.index('</s>')]
                    # BLEUの結果を計算するため.
                    print("".join(src_batch[source_cuont]).replace("</s>", ""))
                    print(' '.join(hyp))
                    print(' '.join(hyp), file=fp)
                    source_cuont = source_cuont + 1

                generated += K

        trace('finished.')

    def print_out(self, K, i_epoch, trained, src_batch, trg_batch, hyp_batch):

            trace('epoch %3d/%3d, sample %8d' % (i_epoch + 1, self.epoch, trained + K + 1))
            trace('  src = ' + ' '.join([x if x != '</s>' else '*' for x in src_batch[K]]))
            trace('  trg = ' + ' '.join([x if x != '</s>' else '*' for x in trg_batch[K]]))
            trace('  hyp = ' + ' '.join([x if x != '</s>' else '*' for x in hyp_batch[K]]))
