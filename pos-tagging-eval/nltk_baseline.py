#!/usr/bin/env python3

import argparse
import logging

from nltk.tag import UnigramTagger, hmm, CRFTagger
from nltk.tag.perceptron import PerceptronTagger

from pos_tagger import UD_HOME, CODE_TO_NAME, read_conllu


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def zip_train_data(train_sentences, train_tag_sequences):
    zipped = []
    for sentence, tags in zip(train_sentences, train_tag_sequences):
        zipped.append(list(zip(sentence, tags)))
    return zipped


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('lang', help='language code')
    args = parser.parse_args()

    with open(f"{UD_HOME}/{CODE_TO_NAME[args.lang]}-train.conllu") as f:
        train_sentences, train_tag_sequences = read_conllu(f)
    #with open(f"{UD_HOME}/{CODE_TO_NAME[args.lang]}-dev.conllu") as f:
    #    dev_sentences, dev_tag_sequences = read_conllu(f)
    with open(f"{UD_HOME}/{CODE_TO_NAME[args.lang]}-test.conllu") as f:
        test_sentences, test_tag_sequences = read_conllu(f)

    logging.info("Training unigram tagger.")
    tagger = UnigramTagger(zip_train_data(train_sentences, train_tag_sequences))

    logging.info("Evaluating unigram tagger.")
    acc = tagger.accuracy(zip_train_data(test_sentences, test_tag_sequences))
    logging.info("Test accuracy: %.2f%%", 100 * acc)

    #logging.info("Training HMM tagger.")
    #tagger = hmm.HiddenMarkovModelTagger.train(zip_train_data(train_sentences, train_tag_sequences))
    #logging.info("Evaluating HMM tagger.")
    #acc = tagger.accuracy(zip_train_data(test_sentences, test_tag_sequences))
    #logging.info("Test accuracy: %.2f%%", 100 * acc)

    logging.info("Training Averaged Perceptron tagger.")
    tagger = PerceptronTagger(load=False)
    tagger.train(zip_train_data(train_sentences, train_tag_sequences))
    logging.info("Evaluating Averaged Perceptron tagger.")
    acc = tagger.accuracy(zip_train_data(test_sentences, test_tag_sequences))
    logging.info("Test accuracy: %.2f%%", 100 * acc)

    logging.info("Done.")


if __name__ == '__main__':
    main()
