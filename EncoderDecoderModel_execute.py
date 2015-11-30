#!/usr/bin/env python
#-*- coding:utf-8 -*-
#!/usr/bin/python3

import util.generators as gens
from util.functions import trace, fill_batch
from util.model_file import ModelFile
from util.vocabulary import Vocabulary

from util.chainer_cpu_wrapper import wrapper

#import pyximport
#pyximport.install()
#from EncoderDecoderModel import EncoderDecoderModel
import EncoderDecoderModelForward

class EncoderDecoderModel_execute:

    def __init__(self, parameter_dict):
        self.source           = parameter_dict["source"]
        self.target           = parameter_dict["target"]
        self.vocab            = parameter_dict["vocab"]
        self.embed            = parameter_dict["embed"]
        self.hidden           = parameter_dict["hidden"] 
        self.epoch            = parameter_dict["epoch"] 
        self.minibatch        = parameter_dict["minibatch"] 
        self.generation_limit = parameter_dict["generation_limit"] 

    def train_model(self):
        trace('making vocaburaries ...')
        src_vocab = Vocabulary.new(gens.word_list(self.source), self.vocab)
        trg_vocab = Vocabulary.new(gens.word_list(self.target), self.vocab)

        trace('making model ...')
        model = EncoderDecoderModel.new(src_vocab, trg_vocab, self.embed, self.hidden)

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

                for k in range(K):
                    trace('epoch %3d/%3d, sample %8d' % (i_epoch + 1, self.epoch, trained + k + 1))
                    trace('  src = ' + ' '.join([x if x != '</s>' else '*' for x in src_batch[k]]))
                    trace('  trg = ' + ' '.join([x if x != '</s>' else '*' for x in trg_batch[k]]))
                    trace('  hyp = ' + ' '.join([x if x != '</s>' else '*' for x in hyp_batch[k]]))

                trained += K

            trace('saving model ...')
            model.save(model + '.%03d' % (self.epoch + 1))

        trace('finished.')

    def test_model(self):
        trace('loading model ...')
        model = EncoderDecoderModel.load(model)
    
        trace('generating translation ...')
        generated = 0

        with open(self.target, 'w') as fp:
            for src_batch in gens.batch(gens.word_list(self.source), self.minibatch):
                src_batch = fill_batch(src_batch)
                K = len(src_batch)

                trace('sample %8d - %8d ...' % (generated + 1, generated + K))
                hyp_batch = model.predict(src_batch, self.generation_limit)

                for hyp in hyp_batch:
                    hyp.append('</s>')
                    hyp = hyp[:hyp.index('</s>')]
                    print(' '.join(hyp), file=fp)

                generated += K

        trace('finished.')
