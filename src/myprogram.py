#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ngramconstructor import NgramConstructor
import math


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    ngrams = NgramConstructor()

    @classmethod
    def load_training_data(cls):
        # our training data is too large to upload to git so we have saved the
        # ngram hash tables to our work directory
        # the NgramConstructor is our way to receive this data, so there's no need
        # to load anything here
        return

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, work_dir):
        # your code here
        lambda_options = [
            [0.1, 0.3, 0.6],
            [0.15, 0.2, 0.7],
            [0.25, 0.25, 0.5],
            [0.3, 0.4, 0.3],
            [0.33, 0.33, 0.34]
        ]

        best_lambdas = []
        best_dev_p = math.inf

        for lambda_ in lambda_options:
            l1, l2, l3 = lambda_
            dev_p = self.ngrams.calculate_dev_perplexity(l1, l2, l3)

            if dev_p < best_dev_p:
                best_dev_p = dev_p
                best_lambdas = lambda_

        # store lambdas in work
        f = open(os.path.join(work_dir, 'trained_lambda.txt'), 'w')
        f.write(str(best_lambdas))
        f.close()

        print("best dev p:", best_dev_p)
        print("best lambda:", best_lambdas)
        print("saved to:", os.path.join(work_dir, 'trained_lambda.txt'))


    def run_pred(self, data):
        # your code here
        preds = []

        START = "<s>"

        # load trained best lambdas 
        with open("work/trained_lambda.txt") as f:
            l1, l2, l3 = eval(f.read())

        vocab = self.ngrams.get_vocab()
        uni_prob = self.ngrams.unigram_prob_train
        bi_prob = self.ngrams.bigram_prob_train
        tri_prob = self.ngrams.trigram_prob_train

        for inp in data:
            # add start <s> symbols as needed
            if len(inp) == 0:
                c1, c2 = START, START 
            elif len(inp) == 1:
                c1, c2 = START, inp[-1]
            else:
                c1, c2 = inp[-2], inp[-1]

            scores = {}
            for c3 in vocab:
                uni = l1 * uni_prob.get(c3, 0)

                bi = 0
                if c2 in vocab:
                    bi = l2 * bi_prob.get(c2+c3, 0)

                tri = 0
                if c1 in vocab and c2 in vocab:
                    tri = l3 * tri_prob.get(c1+c2+c3, 0)

                scores[c3] = uni + bi + tri

            
            top3 = sorted(scores, key=scores.get, reverse=True)[:3]
            preds.append("".join(top3))

        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return MyModel()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        model = MyModel()
        model.run_train(args.work_dir)

    elif args.mode == 'test':
        model = MyModel.load(args.work_dir)
        test_data = MyModel.load_test_data(args.test_data)
        preds = model.run_pred(test_data)
        model.write_pred(preds, args.test_output)