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

from util.chainer_cpu_wrapper import wrapper
#from util.chainer_gpu_wrapper import wrapper

   
class EncoderDecoderModel:

    def __init__(self):
        pass

    def __make_model(self):
        self.__model = wrapper.make_model(
            # encoder
            w_xi = functions.EmbedID(len(self.__src_vocab), self.__n_embed),
            w_ip = functions.Linear(self.__n_embed, 4 * self.__n_hidden),
            w_pp = functions.Linear(self.__n_hidden, 4 * self.__n_hidden),
            # decoder
            w_pq = functions.Linear(self.__n_hidden, 4 * self.__n_hidden),
            w_qj = functions.Linear(self.__n_hidden, self.__n_embed),
            w_jy = functions.Linear(self.__n_embed, len(self.__trg_vocab)),
            w_yq = functions.EmbedID(len(self.__trg_vocab), 4 * self.__n_hidden),
            w_qq = functions.Linear(self.__n_hidden, 4 * self.__n_hidden),
        )

    @staticmethod
    def new(src_vocab, trg_vocab, n_embed, n_hidden):
        self = EncoderDecoderModel()
        self.__src_vocab = src_vocab
        self.__trg_vocab = trg_vocab
        self.__n_embed = n_embed
        self.__n_hidden = n_hidden
        self.__make_model()
        return self

    def save(self, filename):
        with ModelFile(filename, 'w') as fp:
            self.__src_vocab.save(fp.get_file_pointer())
            self.__trg_vocab.save(fp.get_file_pointer())
            fp.write(self.__n_embed)
            fp.write(self.__n_hidden)
            wrapper.begin_model_access(self.__model)
            fp.write_embed(self.__model.w_xi)
            fp.write_linear(self.__model.w_ip)
            fp.write_linear(self.__model.w_pp)
            fp.write_linear(self.__model.w_pq)
            fp.write_linear(self.__model.w_qj)
            fp.write_linear(self.__model.w_jy)
            fp.write_embed(self.__model.w_yq)
            fp.write_linear(self.__model.w_qq)
            wrapper.end_model_access(self.__model)

    @staticmethod
    def load(filename):
        self = EncoderDecoderModel()
        with ModelFile(filename) as fp:
            self.__src_vocab = Vocabulary.load(fp.get_file_pointer())
            self.__trg_vocab = Vocabulary.load(fp.get_file_pointer())
            self.__n_embed = int(fp.read())
            self.__n_hidden = int(fp.read())
            self.__make_model()
            wrapper.begin_model_access(self.__model)
            fp.read_embed(self.__model.w_xi)
            fp.read_linear(self.__model.w_ip)
            fp.read_linear(self.__model.w_pp)
            fp.read_linear(self.__model.w_pq)
            fp.read_linear(self.__model.w_qj)
            fp.read_linear(self.__model.w_jy)
            fp.read_embed(self.__model.w_yq)
            fp.read_linear(self.__model.w_qq)
            wrapper.end_model_access(self.__model)
        return self

    def init_optimizer(self):
        self.__opt = optimizers.AdaGrad(lr=0.01)
        self.__opt.setup(self.__model)

    def __forward(self, is_training, src_batch, trg_batch = None, generation_limit = None):
        m = self.__model
        tanh = functions.tanh
        lstm = functions.lstm
        batch_size = len(src_batch)
        src_len = len(src_batch[0])
        src_stoi = self.__src_vocab.stoi
        trg_stoi = self.__trg_vocab.stoi
        trg_itos = self.__trg_vocab.itos
        s_c = wrapper.zeros((batch_size, self.__n_hidden))
        
        # encoding
        s_x = wrapper.make_var([src_stoi('</s>') for _ in range(batch_size)], dtype=np.int32)
        s_i = tanh(m.w_xi(s_x))
        s_c, s_p = lstm(s_c, m.w_ip(s_i))

        for l in reversed(range(src_len)):
            s_x = wrapper.make_var([src_stoi(src_batch[k][l]) for k in range(batch_size)], dtype=np.int32)
            s_i = tanh(m.w_xi(s_x))
            s_c, s_p = lstm(s_c, m.w_ip(s_i) + m.w_pp(s_p))

        s_c, s_q = lstm(s_c, m.w_pq(s_p))
        hyp_batch = [[] for _ in range(batch_size)]
        
        # decoding
        if is_training:
            accum_loss = wrapper.zeros(())
            trg_len = len(trg_batch[0])
            
            for l in range(trg_len):
                s_j = tanh(m.w_qj(s_q))
                r_y = m.w_jy(s_j)
                s_t = wrapper.make_var([trg_stoi(trg_batch[k][l]) for k in range(batch_size)], dtype=np.int32)
                accum_loss += functions.softmax_cross_entropy(r_y, s_t)
                output = wrapper.get_data(r_y).argmax(1)

                for k in range(batch_size):
                    hyp_batch[k].append(trg_itos(output[k]))

                s_c, s_q = lstm(s_c, m.w_yq(s_t) + m.w_qq(s_q))

            return hyp_batch, accum_loss
        else:
            while len(hyp_batch[0]) < generation_limit:
                s_j = tanh(m.w_qj(s_q))
                r_y = m.w_jy(s_j)
                output = wrapper.get_data(r_y).argmax(1)

                for k in range(batch_size):
                    hyp_batch[k].append(trg_itos(output[k]))

                if all(hyp_batch[k][-1] == '</s>' for k in range(batch_size)): break

                s_y = wrapper.make_var(output, dtype=np.int32)
                s_c, s_q = lstm(s_c, m.w_yq(s_y) + m.w_qq(s_q))
            
            return hyp_batch

    def train(self, src_batch, trg_batch):
        self.__opt.zero_grads()
        hyp_batch, accum_loss = self.__forward(True, src_batch, trg_batch=trg_batch)
        accum_loss.backward()
        self.__opt.clip_grads(10)
        self.__opt.update()
        return hyp_batch

    def predict(self, src_batch, generation_limit):
        return self.__forward(False, src_batch, generation_limit=generation_limit)

